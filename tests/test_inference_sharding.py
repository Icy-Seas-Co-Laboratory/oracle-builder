from __future__ import annotations
import uuid
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit
from oracle_builder.orchestration.inference_sharding import InferenceShardPlan, infer_shard_unit, merge_unit, next_chain_step

def _unit():
 return WorkUnit(work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()), action="infer", inputs={"input":ArtifactRef("dataset","data"),"model":ArtifactRef("model_run","model")}, configuration=None, staging=ArtifactRef("staging","stage"), resources={}, parameters={"split":"all"})

def test_ordered_plan_and_units_are_item_id_stable():
 p=InferenceShardPlan(("c","a","b"), shard_size=2, unsharded_threshold=1)
 assert p.shards()==(("a","b"),("c",))
 unit=infer_shard_unit(_unit(), shard_id="shard-00000", item_ids=("b","a"))
 assert unit.parameters["shard_item_ids"]==["a","b"]
 assert next_chain_step(p,{})==( "shard-00000", ("a","b"))
 merged=merge_unit(_unit(), expected_item_ids=p.item_ids, shard_outputs={"shard-00000":ArtifactRef("inference_result","one")})
 assert merged.parameters["merge_item_ids"]==["a","b","c"] and "shard_shard-00000" in merged.inputs


def test_server_validates_shard_rows_and_input_identity(tmp_path):
 import json, sqlite3, pytest
 from oracle_builder.orchestration.inference_sharding import InferenceShardingMixin, item_ids_digest
 unit = infer_shard_unit(_unit(), shard_id='shard-00000', item_ids=['a','b'])
 artifact = {'model': {'artifact_id': 'model'}, 'input': {'reference': unit.inputs['input'].to_dict()}, 'parameters': {'prediction_set':'predictions'}}
 (tmp_path/'artifact.json').write_text(json.dumps(artifact))
 (tmp_path/'inference_shard.json').write_text(json.dumps({'schema':{'name':'oracle_builder_inference_shard','version':1}, 'shard_id':'shard-00000','item_ids':['a','b'],'item_ids_sha256':item_ids_digest(['a','b'])}))
 with sqlite3.connect(tmp_path/'predictions.sqlite') as db:
  db.execute('CREATE TABLE predictions(uuid TEXT, prediction_set TEXT)')
  db.executemany('INSERT INTO predictions VALUES(?,?)', [('a','predictions'),('b','predictions')])
 validator = InferenceShardingMixin()
 validator._validate_inference_shard_output(tmp_path, unit)
 artifact['input']['reference']['artifact_id'] = 'different'
 (tmp_path/'artifact.json').write_text(json.dumps(artifact))
 with pytest.raises(ValueError, match='input identity'):
  validator._validate_inference_shard_output(tmp_path, unit)
 artifact['input']['reference'] = unit.inputs['input'].to_dict()
 (tmp_path/'artifact.json').write_text(json.dumps(artifact))
 with sqlite3.connect(tmp_path/'predictions.sqlite') as db: db.execute("DELETE FROM predictions WHERE uuid='b'")
 with pytest.raises(ValueError, match='prediction rows'):
  validator._validate_inference_shard_output(tmp_path, unit)
 merge = merge_unit(_unit(), expected_item_ids=['a','b'], shard_outputs={'shard-00000':ArtifactRef('inference_result','one')})
 with pytest.raises(ValueError, match='prediction rows'):
  validator._validate_inference_merge_output(tmp_path, merge)
