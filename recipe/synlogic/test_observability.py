"""CPU integration checks for diagnostic isolation and replayable rollout snapshots."""
from copy import deepcopy
import gzip, json, os, pickle, random, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from synlogic.observability import RolloutObserver, write_snapshot, verifier_classes

class Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return {1:'unparseable',2:'<answer>good</answer>',3:'<answer>bad</answer>'}.get(int(ids[0]),'prompt')
class Verifier:
    def extract_answer(self, answer):return answer
    def verify(self, game, answer):
        random.random();np.random.rand();torch.rand(1)
        if game.answer=='ERROR':raise RuntimeError('Synthetic verifier failure')
        result=game.answer==answer;game.answer='mutated diagnostic copy';return result

def state():return pickle.dumps((random.getstate(),np.random.get_state(),torch.random.get_rng_state().numpy()))

import unittest
from unittest.mock import patch

class ObservabilityTests(unittest.TestCase):
    def test_complete_read_only_observation_contract(self):
        self.assertEqual(torch.cuda.device_count(),0)
        with patch.dict(os.environ), patch.dict(verifier_classes, {"__observation_test__":Verifier}):
            with tempfile.TemporaryDirectory() as temporary:
                os.environ.update(SYNLOGIC_RUN_ROOT=temporary,SYNLOGIC_ALLOCATION_ID='cpu-test',SYNLOGIC_ATTEMPT='1')
                Path(temporary,'portable-provenance.json').write_text(json.dumps({'protocol_id':'test-baseprompt',
                    'source_revision':{'test':'recorded'},'data_fingerprint':{'test':'data'}}))
                trainer=SimpleNamespace(config=OmegaConf.create({'trainer':{'default_local_dir':temporary+'/checkpoints',
                    'test_freq':20,'total_training_steps':440},'actor_rollout_ref':{'rollout':{'n':4}},
                    'data':{'train_batch_size':2},'provenance':{'protocol_id':'test-baseprompt'}}),
                    global_steps=19,_completed_updates=18,reward_fn=SimpleNamespace(reward=SimpleNamespace(tokenizer=Tokenizer())))
                observer=RolloutObserver(trainer)
                originals=[]
                for i,uid in enumerate(['g0','g1']):
                    extra={'index':i,'target_mode':'fixed_wrong_answer','assigned_answer_key':'bad',
                           'game_data_str':json.dumps({'question':'q','answer':'good','metadata':{}})}
                    originals.append(deepcopy(extra));observer.freeze(19,uid,'__observation_test__',extra,{'ground_truth':'bad'},extra,{'ground_truth':'bad'})
                    extra['assigned_answer_key']='changed after freeze'
                responses=torch.tensor([[3,0],[3,0],[2,0],[1,0],[2,0],[2,0],[2,0],[2,0]])
                prompt=torch.full((8,2),9);mask=torch.tensor([[1,0]]*8)
                advantages=torch.zeros(8,2);advantages[2,0]=1;advantages[4,1]=7
                tensors={'prompts':prompt,'responses':responses,'input_ids':torch.cat([prompt,responses],-1),
                    'attention_mask':torch.cat([torch.ones_like(prompt),mask],-1),'response_mask':mask,
                    'position_ids':torch.arange(4).repeat(8,1),'advantages':advantages,
                    'token_level_scores':torch.zeros(8,2),'token_level_rewards':torch.zeros(8,2),
                    'old_log_probs':torch.zeros(8,2),'ref_log_prob':torch.zeros(8,2)}
                batch=DataProto.from_dict(tensors=tensors,non_tensors={'uid':np.array(['g0']*4+['g1']*4,dtype=object)})
                batch.reorder(torch.tensor([4,0,5,1,6,2,7,3]))
                before={k:v.clone() for k,v in batch.batch.items()};rng=state();trainer._completed_updates=19
                metrics={};observer.observe(batch,metrics);observer.close()
                assert state()==rng,'Diagnostics advanced training RNG'
                assert all(torch.equal(v,batch.batch[k]) for k,v in before.items()),'Diagnostics changed optimization data'
                assert metrics['noise/noisy_question_group_active_rate']==.5
                assert metrics['noise/noisy_rollout_assigned_label_match_rate']==.25
                assert metrics['noise/noisy_rollout_original_verifier_correctness_rate']==.625
                folder=next(observer.root.glob('step_*'));manifest=json.loads((folder/'manifest.json').read_text())
                assert manifest['complete'] and manifest['rows']==8 and manifest['groups']==2
                assert manifest['rollout_policy_version']==18 and manifest['update_version']==19
                with gzip.open(folder/'rollouts.pt.gz','rb') as stream:payload=torch.load(stream,weights_only=False)
                assert len({row['rollout_id'] for row in payload['rows']})==8
                assert all(row['frozen_assigned_label']['ground_truth']=='bad' for row in payload['rows'])
                assert torch.equal(payload['tensors']['advantages'],before['advantages'])
                assert write_snapshot(observer.root,payload)['file_sha256']==manifest['file_sha256']
                conflicting=deepcopy(payload);conflicting['metadata']['configuration_sha256']='different'
                try:write_snapshot(observer.root,conflicting)
                except RuntimeError:pass
                else:raise AssertionError('Conflicting replay overwrote the original')
                # A clean-only step has counts and explicit undefined flags, with no fake zero rates.
                trainer.global_steps=20;observer=RolloutObserver(trainer)
                for i,uid in enumerate(['g0','g1']):
                    extra={**originals[i],'target_mode':'original_verifier'}
                    observer.freeze(20,uid,'__observation_test__',extra,{},extra,{})
                clean={};observer.observe(batch,clean);observer.close()
                assert clean['noise/noisy_question_groups']==0
                for name in ['noisy_question_group_active_rate','noisy_rollout_assigned_label_match_rate','noisy_rollout_original_verifier_correctness_rate']:
                    assert f'noise/{name}' not in clean
                    assert clean[f'noise/{name}/defined']==0 and clean[f'noise/{name}/denominator']==0
                # Verifier failures are excluded from measured correctness and counted explicitly.
                trainer.global_steps=21;observer=RolloutObserver(trainer)
                for i,uid in enumerate(['g0','g1']):
                    extra=deepcopy(originals[i]);extra['game_data_str']=json.dumps({'question':'q','answer':'ERROR','metadata':{}})
                    observer.freeze(21,uid,'__observation_test__',extra,{},extra,{})
                failed={};observer.observe(batch,failed);observer.close()
                assert failed['noise/original_verifier_failures']==7
                assert failed['noise/noisy_rollout_original_verifier_correctness_rate/denominator']==1
                assert failed['noise/noisy_rollout_original_verifier_correctness_rate']==0
                # Effective OLR replacements must not overwrite the original frozen target.
                observer=RolloutObserver(trainer)
                old=deepcopy(originals[0]);effective={**old,'assigned_answer_key':'refined'}
                observer.freeze(21,'g0','__observation_test__',old,{'ground_truth':'bad'},effective,{'ground_truth':'refined'})
                effective['assigned_answer_key']='mutated after capture'
                assert observer.frames['g0']['frozen_assigned_label']['ground_truth']=='bad'
                assert observer.frames['g0']['effective_optimization_label']['ground_truth']=='refined'
                assert observer.frames['g0']['original_extra']['assigned_answer_key']=='bad'
                assert observer.frames['g0']['effective_optimization_extra']['assigned_answer_key']=='refined'
                observer.close()

if __name__=="__main__":unittest.main()
