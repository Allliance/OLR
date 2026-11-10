"""Read-only rollout diagnostics and immutable pre-evaluation snapshots."""
from collections import Counter, defaultdict
from fractions import Fraction
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

import numpy as np
import torch

from src.fixed_labels import answer_key, extract_answer
from src.syn_logic_reward import Data, TaskDeadlineExceeded, verifier_classes, verifier_deadline


def mathador_true_correct(data, extracted, verifier):
    """Shadow-only correction of upstream's overwritten target/self-comparison.

    The optimization verifier is untouched. Evaluate arithmetic without eval().
    """
    expression = verifier.extract_answer(extracted)
    if not expression or len(expression) > 512:
        return False
    metadata = data.metadata
    if Counter(re.findall(r'\d+', expression)) != Counter(str(n) for n in metadata['numbers']):
        return False
    if any(c not in '0123456789() \n\t' and c not in metadata['operators'] for c in expression):
        return False
    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
        raise ValueError('Not a permitted arithmetic expression')
    try:
        value = evaluate(ast.parse(expression, mode='eval').body)
        return abs(value - Fraction(str(metadata['result']))) < Fraction(1, 10**10)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return False


def snapshot_due(step, frequency, target):
    next_step = step + 1
    return frequency > 0 and next_step <= target and (next_step % frequency == 0 or next_step == target)


def plain(value):
    if isinstance(value, dict):return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):return [plain(v) for v in value]
    if isinstance(value, np.generic):return value.item()
    if hasattr(value, 'to_json'):return plain(value.to_json())
    if value is None or isinstance(value, (str, int, float, bool)):return value
    raise TypeError(f'Unsupported observation metadata: {type(value).__name__}')


def encoded(value):
    return json.dumps(plain(value), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):h.update(block)
    return h.hexdigest()


def configuration_without_credentials(config):
    from omegaconf import OmegaConf
    def check(value):
        if isinstance(value,dict):
            for key,item in value.items():
                if str(key).lower() in {'api_key','token','password','authorization','secret','access_token','hf_token','wandb_api_key'} and item:
                    raise ValueError('Credentials must stay outside snapshot configuration')
                check(item)
        elif isinstance(value,list):
            for item in value:check(item)
        elif isinstance(value,str) and '${oc.env:' in value:
            raise ValueError('Environment interpolation is not safe snapshot provenance')
    unresolved=OmegaConf.to_container(config,resolve=False);check(unresolved)
    resolved=OmegaConf.to_container(config,resolve=True);check(resolved)
    return resolved


@contextmanager
def diagnostic_rng():
    python=random.getstate();numpy=np.random.get_state();cpu=torch.random.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.random.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)


def rate(numerator, denominator):
    return {'numerator':int(numerator), 'denominator':int(denominator),
            'defined':bool(denominator), 'value':numerator/denominator if denominator else None}


def write_snapshot(root, payload):
    """Publish one complete immutable directory; identical retries are no-ops."""
    root=Path(root);root.mkdir(parents=True, exist_ok=True,mode=0o700)
    assert root.stat().st_uid==os.getuid();root.chmod(0o700)
    identity=payload['metadata'];step=identity['training_step'];execution=identity['execution_id']
    destination=root/f'step_{step:06d}-{execution}'
    digest=hashlib.sha256(encoded({'metadata':identity, 'rows':payload['rows']}))
    for name, tensor in sorted(payload['tensors'].items()):
        digest.update(encoded([name, str(tensor.dtype), list(tensor.shape)]))
        flat=tensor.contiguous().reshape(-1).view(torch.uint8).numpy()
        for start in range(0, flat.size, 8*1024**2):digest.update(memoryview(flat[start:start+8*1024**2]))
    payload_hash=digest.hexdigest()
    if destination.exists():
        old=json.loads((destination/'manifest.json').read_text())
        if old['payload_sha256']!=payload_hash or checksum(destination/'rollouts.pt.gz')!=old['file_sha256']:
            raise RuntimeError('Conflicting or corrupt rollout snapshot; original preserved')
        return old
    temporary=Path(tempfile.mkdtemp(prefix='.snapshot-', dir=root))
    try:
        file=temporary/'rollouts.pt.gz'
        with file.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
                torch.save(payload, compressed)
            raw.flush();os.fsync(raw.fileno())
        manifest={**identity, 'complete':True, 'payload_sha256':payload_hash,
                  'file_sha256':checksum(file), 'compressed_bytes':file.stat().st_size,
                  'rows':len(payload['rows']),
                  'groups':len({r['group_id'] for r in payload['rows']}),
                  'tensor_shapes':{k:list(v.shape) for k,v in payload['tensors'].items()}}
        with (temporary/'manifest.json').open('wb') as stream:
            stream.write(encoded(manifest));stream.flush();os.fsync(stream.fileno())
        descriptor=os.open(temporary,os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        temporary.rename(destination)
        descriptor=os.open(root, os.O_DIRECTORY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        return manifest
    finally:
        if temporary.exists():shutil.rmtree(temporary)


class RolloutObserver:
    def __init__(self, trainer):
        self.trainer=trainer;self.frames={};self.step=None
        self.executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix='rollout-snapshot')
        self.pending=None
        run=Path(os.environ['SYNLOGIC_RUN_ROOT']).resolve()
        assert Path(trainer.config.trainer.default_local_dir).resolve()==run/'checkpoints', 'Run-specific snapshot layout required'
        self.root=run/'rollout_snapshots'
        job=os.environ['SYNLOGIC_ALLOCATION_ID'];attempt=os.environ['SYNLOGIC_ATTEMPT']
        assert job and attempt and re.fullmatch(r'[A-Za-z0-9_.-]+',job) and re.fullmatch(r'[A-Za-z0-9_.-]+',attempt)
        self.execution='job'+job+'-a'+attempt
        self.provenance=json.loads((run/'portable-provenance.json').read_text())
        assert self.provenance['protocol_id']==str(trainer.config.provenance.protocol_id)
        assert self.provenance['source_revision'] and self.provenance['data_fingerprint']
        self.configuration=configuration_without_credentials(trainer.config)
        tracker=run/'checkpoints/latest_checkpointed_iteration.txt'
        self.resumed_from_step=int(tracker.read_text()) if tracker.exists() else 0

    def freeze(self, step, uid, task, original_extra, original_target, effective_extra, effective_target):
        if self.step!=step:
            self.frames={};self.step=step
            self.rollout_policy_version=int(getattr(self.trainer,'_completed_updates',step-1))
        self.frames[str(uid)]={
            'task':str(task), 'question_index':int(original_extra['index']),
            'original_extra':deepcopy(original_extra),
            'frozen_assigned_label':deepcopy(original_target),
            'effective_optimization_extra':deepcopy(effective_extra),
            'effective_optimization_label':deepcopy(effective_target),
            'corrupted':original_extra.get('target_mode')=='fixed_wrong_answer'}

    def _score(self, text, frame):
        """Frozen dataset target versus original native reference, never a reward input."""
        extracted=extract_answer(text)
        if extracted is None:return 0, 0, False, False, True
        try:verifier=verifier_classes[frame['task']]()
        except (Exception, TaskDeadlineExceeded):return None,None,True,True,False
        assigned=native=None;assigned_failed=native_failed=False
        try:
            with verifier_deadline():
                assigned=int(answer_key(extracted, verifier)==frame['original_extra']['assigned_answer_key'])
        except (Exception, TaskDeadlineExceeded):assigned_failed=True
        try:
            extra=frame['original_extra']
            source_key='original_game_data_str' if frame['corrupted'] else 'game_data_str'
            if frame['corrupted'] and source_key not in extra:
                raise ValueError('Missing original clean reference for noisy diagnostic')
            game=Data.from_json_str(extra[source_key]) if source_key in extra else deepcopy(extra['game_data'])
            with verifier_deadline(), open(os.devnull,'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
                native=int(bool(mathador_true_correct(game,extracted,verifier) if frame['task']=='mathador' else verifier.verify(game, extracted)))
        except (Exception, TaskDeadlineExceeded):native_failed=True
        return assigned, native, assigned_failed, native_failed, False

    def observe(self, batch, metrics):
        if self.pending is not None and self.pending.done():self.pending.result()
        step=int(self.trainer.global_steps);assert self.step==step
        uids=[str(x) for x in batch.non_tensor_batch['uid']]
        groups=defaultdict(list)
        for i,uid in enumerate(uids):groups[uid].append(i)
        expected=int(self.trainer.config.actor_rollout_ref.rollout.n)
        expected_groups=int(self.trainer.config.data.train_batch_size)
        assert len(groups)==expected_groups and len(batch)==expected_groups*expected, 'Incomplete training batch'
        assert all(len(indices)==expected for indices in groups.values()), 'Incomplete rollout group'
        assert all(uid in self.frames for uid in groups), 'Missing frozen labels'
        tensors=batch.batch;mask=tensors['response_mask'].bool();advantages=tensors['advantages']
        assert mask.shape==advantages.shape
        noisy={uid for uid in groups if self.frames[uid]['corrupted']}
        active=sum(bool((advantages[indices].ne(0)&mask[indices]).any()) for uid,indices in groups.items() if uid in noisy)
        assigned_n=assigned_d=native_n=native_d=assigned_fail=native_fail=parse_fail=0
        diagnostic_rows={};tokenizer=self.trainer.reward_fn.reward.tokenizer
        with diagnostic_rng():
            for uid in noisy:
                for i in groups[uid]:
                    ids=tensors['responses'][i][mask[i]].tolist()
                    text=tokenizer.decode(ids,skip_special_tokens=True)
                    a,n,af,nf,pf=self._score(text,self.frames[uid])
                    assigned_fail+=af;native_fail+=nf;parse_fail+=pf
                    if not af:assigned_d+=1;assigned_n+=a
                    if not nf:native_d+=1;native_n+=n
                    diagnostic_rows[i]={'frozen_assigned_match':a,'original_verifier_correct':n,
                                        'assigned_verifier_failure':af,'original_verifier_failure':nf,'parse_failure':pf}
        noisy_rollouts=sum(len(groups[uid]) for uid in noisy)
        diagnostics={
            'noisy_question_group_active_rate':rate(active,len(noisy)),
            'noisy_rollout_assigned_label_match_rate':rate(assigned_n,assigned_d),
            'noisy_rollout_original_verifier_correctness_rate_v2':rate(native_n,native_d),
            'assigned_label_scoring_coverage':rate(assigned_d,noisy_rollouts),
            'original_reference_scoring_coverage':rate(native_d,noisy_rollouts),
            'original_verifier_success_coverage':rate(native_d-parse_fail,noisy_rollouts)}
        for name,value in diagnostics.items():
            for field in ['numerator','denominator','defined']:metrics[f'noise/{name}/{field}']=int(value[field])
            if value['defined']:metrics[f'noise/{name}']=value['value']
        metrics.update({'noise/question_groups':len(groups),'noise/noisy_question_groups':len(noisy),
                        'noise/noisy_rollouts':noisy_rollouts,'noise/assigned_verifier_failures':assigned_fail,
                        'noise/original_verifier_failures':native_fail,'noise/parse_failures':parse_fail,
                        'noise/original_verifier_successes':native_d-parse_fail,
                        'noise/original_reference_evaluated_rollouts':native_d})
        frequency=int(self.trainer.config.trainer.test_freq)
        target=int(getattr(self.trainer,'total_training_steps',None) or self.trainer.config.trainer.total_training_steps)
        save=snapshot_due(step, frequency, target)
        metrics['observability/snapshot_scheduled']=int(save)
        if not save:return
        if self.pending is not None:self.pending.result()
        configuration=self.configuration;provenance=self.provenance
        rows=[];occurrences=Counter()
        for i,uid in enumerate(uids):
            frame=self.frames[uid];slot=occurrences[uid];occurrences[uid]+=1
            prompt_mask=tensors['attention_mask'][i,:tensors['prompts'].shape[-1]].bool()
            prompt_ids=tensors['prompts'][i][prompt_mask].tolist();response_ids=tensors['responses'][i][mask[i]].tolist()
            question=f"{frame['task']}:{frame['question_index']}"
            group=f"{self.execution}:{step}:{uid}"
            rows.append({'row':i,'question_id':question,'group_id':group,'rollout_id':f'{group}:{slot}',
                'source_group_uid':uid,'prompt_text':tokenizer.decode(prompt_ids,skip_special_tokens=False),
                'response_text':tokenizer.decode(response_ids,skip_special_tokens=False),
                'prompt_token_ids':prompt_ids,'response_token_ids':response_ids,
                'frozen_assigned_label':frame['frozen_assigned_label'],
                'original_clean_reference':{
                    'task':frame['task'],'index':frame['question_index'],
                    'source_field':'original_game_data_str' if frame['corrupted'] else 'game_data_str',
                    'game_data_str':frame['original_extra'].get('original_game_data_str' if frame['corrupted'] else 'game_data_str'),
                    'available':isinstance(frame['original_extra'].get('original_game_data_str' if frame['corrupted'] else 'game_data_str'),str)},
                'effective_optimization_label':frame['effective_optimization_label'],
                'effective_optimization_extra':frame['effective_optimization_extra'],
                'originally_corrupted':frame['corrupted'],'diagnostics':diagnostic_rows.get(i)})
        assert len(rows)==len(batch) and len({r['rollout_id'] for r in rows})==len(rows)
        metadata={'schema_version':2,'execution_id':self.execution,'training_step':step,
            'update_version':int(getattr(self.trainer,'_completed_updates',step)),
            'rollout_policy_version':self.rollout_policy_version,'evaluation_step':step+1,
            'resumed_from_step':self.resumed_from_step,
            'protocol_id':str(self.trainer.config.provenance.protocol_id),
            'expected_rollouts':expected_groups*expected,'expected_groups':expected_groups,'rollouts_per_group':expected,
            'configuration':configuration,'configuration_sha256':hashlib.sha256(encoded(configuration)).hexdigest(),
            'source_provenance':provenance,'diagnostics':diagnostics,
            'assigned_match_definition':'Frozen dataset assigned target, before OLR replacement',
            'original_reference_definition':'For noisy rows use only original_game_data_str; missing originals are verifier failures, never fallback to corrupted game data. Mathador arithmetic uses a read-only target check because the upstream verifier compares the evaluated result to itself.',
            'scoring_coverage_definition':'Unparseable responses are known incorrect; verifier errors are unknown. Native verifier successes exclude parse failures.',
            'active_definition':'Any exactly nonzero advantage at a valid response token within a noisy group'}
        payload={'metadata':plain(metadata),'rows':plain(rows),
                 'tensors':{key:value.detach().cpu().clone() for key,value in tensors.items()}}
        required={'prompts','responses','attention_mask','response_mask','token_level_scores','token_level_rewards','advantages'}
        assert required<=payload['tensors'].keys(), 'Missing optimization tensors'
        self.pending=self.executor.submit(write_snapshot,self.root,payload)

    def close(self):
        try:
            if self.pending is not None:self.pending.result()
        finally:self.executor.shutdown(wait=True)

    def flush(self):
        if self.pending is not None:self.pending.result()
