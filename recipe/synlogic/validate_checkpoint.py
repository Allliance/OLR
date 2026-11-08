"""CPU validation of every model/optimizer/RNG shard of a SynLogic checkpoint."""
import argparse,ctypes,fcntl,json,math,mmap,os,re,subprocess,sys
from pathlib import Path

def require(value,message):
    if not value:raise ValueError(message)

def worker(folder):
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    torch.set_num_threads(2);dist.init_process_group('gloo')
    rank=dist.get_rank();world=dist.get_world_size();step=int(folder.name.removeprefix('global_step_'))
    require(torch.cuda.device_count()==0,'Validation must remain CPU-only')
    require(json.loads((folder/'complete.json').read_text())['step']==step,'Incomplete checkpoint')
    state=json.loads((folder/'olr_state.json').read_text())
    from template import PINNED_TEMPLATE_SHA256
    from refinement import OnlineLabelRefiner
    require(state['prompt_template_sha256']==PINNED_TEMPLATE_SHA256,'Template mismatch')
    require(state['step']==state['last_reward_step']==step,'Model/reward state mismatch')
    refiner=OnlineLabelRefiner(**state['refiner']['config']);refiner.load_state_dict(state['refiner'])
    visits=sum(len(history['rates']) for history in refiner.history.values())
    require(visits==step*160,'Incomplete processed-prompt history')
    if 'oracle-clean' in state['protocol_id']:
        require(not state['refiner']['config']['enabled'],'Oracle-discard baseline must disable OLR')
    def pieces(value):
        if isinstance(value,DTensor):return [value.to_local()]
        if isinstance(value,ShardedTensor):return [s.tensor for s in value.local_shards()]
        return [value] if isinstance(value,torch.Tensor) else []
    def finite(value):
        for tensor in pieces(value):
            if tensor.is_floating_point():
                flat=tensor.reshape(-1)
                for start in range(0,flat.numel(),1_000_000):
                    if not bool(torch.isfinite(flat[start:start+1_000_000]).all()):return False
        return True
    original=torch.UntypedStorage.from_file;libc=ctypes.CDLL(None,use_errno=True)
    libc.mprotect.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];libc.mprotect.restype=ctypes.c_int
    def readonly(*args,**kwargs):
        storage=original(*args,**kwargs)
        if storage.nbytes() and libc.mprotect(storage.data_ptr(),storage.nbytes(),mmap.PROT_READ):
            raise OSError(ctypes.get_errno(),'Cannot protect checkpoint mapping')
        return storage
    torch.UntypedStorage.from_file=readonly
    files=[folder/f'actor/{kind}_world_size_{world}_rank_{rank}.pt' for kind in ['model','optim','extra_state']]
    signatures=[(p.stat().st_size,p.stat().st_mtime_ns) for p in files]
    with torch.serialization.set_default_mmap_options(mmap.MAP_SHARED):
        model=torch.load(files[0],map_location='cpu',weights_only=False,mmap=True)
        shapes={k:tuple(v.size()) for k,v in model.items() if isinstance(v,(torch.Tensor,ShardedTensor))}
        parameters=sum(math.prod(shape) for shape in shapes.values())
        require(3_900_000_000<=parameters<=4_600_000_000,'Incomplete Qwen3-4B parameter set')
        require(all(any(f'layers.{i}.' in key for key in model) for i in range(36)),'Missing model layers')
        require(all(finite(v) for v in model.values()),'Non-finite model tensor')
        local_elements=sum(t.numel() for v in model.values() for t in pieces(v));del model
        optimizer=torch.load(files[1],map_location='cpu',weights_only=False,mmap=True)
        require(optimizer['state'] and optimizer['param_groups'],'Missing optimizer')
        require(all(math.isfinite(g['lr']) and g['lr']>0 for g in optimizer['param_groups']),'Invalid LR')
        require(all(float(v['step'])==5*step for v in optimizer['state'].values()),'Wrong optimizer counters')
        require(all(finite(t) for v in optimizer['state'].values() for t in v.values()),'Non-finite optimizer')
        optimizer_states=len(optimizer['state']);del optimizer
    extra=torch.load(files[2],map_location='cpu',weights_only=False)
    require(extra.get('lr_scheduler') is not None and extra.get('rng'),'Missing scheduler/RNG')
    require(signatures==[(p.stat().st_size,p.stat().st_mtime_ns) for p in files],'Checkpoint changed during validation')
    payload={'rank':rank,'parameters':parameters,'local_model_elements':local_elements,
             'optimizer_states':optimizer_states,'shapes':shapes,'all_tensors_finite':True}
    proofs=[None]*world;dist.all_gather_object(proofs,payload)
    if rank==0:
        require(all(p['shapes']==proofs[0]['shapes'] for p in proofs),'Rank parameter sets disagree')
        loader=torch.load(folder/'data.pt',map_location='cpu',weights_only=False)
        require(isinstance(loader,dict) and loader and loader.get('_synlogic_sampler'),'Missing resumable sampler')
        for proof in proofs:proof.pop('shapes')
        result={'step':step,'validated':True,'world_size':world,'model_parameters':parameters,
            'protocol_id':state['protocol_id'],'prompt_template_sha256':PINNED_TEMPLATE_SHA256,
            'optimizer_steps':5*step,'olr_prompt_visits':visits,'rank_proofs':proofs,
            'all_model_optimizer_tensors_finite':True,
            'components':['model','optimizer','lr_scheduler','rng','dataloader','sampler','olr_history']}
        tmp=folder/'validated.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(folder/'validated.json')
        print(json.dumps(result),flush=True)
    dist.barrier();dist.destroy_process_group()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:return worker(args.checkpoint)
    folder=args.checkpoint;lock=(folder.parent/('.validation-'+folder.name+'.lock')).open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    files=list((folder/'actor').glob('model_world_size_*_rank_*.pt'));require(files,'Missing model shards')
    worlds={int(re.search(r'world_size_(\d+)_rank_',p.name)[1]) for p in files};require(len(worlds)==1,'Mixed world sizes');world=worlds.pop()
    require(len(files)==world,'Missing model rank')
    marker=folder/'validated.json'
    if marker.exists():
        proof=json.loads(marker.read_text());require(proof['validated'] and proof.get('world_size',1)==world,'Invalid existing proof');return
    env={**os.environ,'ROCR_VISIBLE_DEVICES':'','HIP_VISIBLE_DEVICES':'','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'}
    subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',str(Path(__file__).resolve()),str(folder),'--worker'],env=env,check=True,timeout=900)

if __name__=='__main__':main()
