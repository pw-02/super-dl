# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: superdl/syncgrpc/protos/cache_coordinator.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/superdl/syncgrpc/protos/cache_coordinator.proto\x1a\x1bgoogle/protobuf/empty.proto\">\n\x13RegisterJobResponse\x12\x16\n\x0ejob_registered\x18\x01 \x01(\x08\x12\x0f\n\x07message\x18\x02 \x01(\t\"c\n\x0fRegisterJobInfo\x12\x0e\n\x06job_id\x18\x01 \x01(\x05\x12\x12\n\ndataset_id\x18\x02 \x01(\t\x12\x10\n\x08\x64\x61ta_dir\x18\x03 \x01(\t\x12\x1a\n\x12\x64\x61ta_source_system\x18\x04 \x01(\t\"U\n\x16\x42\x61tchAccessPatternList\x12\x0e\n\x06job_id\x18\x01 \x01(\x03\x12\x17\n\x07\x62\x61tches\x18\x02 \x03(\x0b\x32\x06.Batch\x12\x12\n\nbatch_type\x18\x03 \x01(\t\"1\n\x05\x42\x61tch\x12\x10\n\x08\x62\x61tch_id\x18\x01 \x01(\x03\x12\x16\n\x0esample_indices\x18\x02 \x03(\x05\"\x1a\n\x07Message\x12\x0f\n\x07message\x18\x01 \x01(\t2\x9c\x01\n\x17\x43\x61\x63heCoordinatorService\x12\x35\n\x0bRegisterJob\x12\x10.RegisterJobInfo\x1a\x14.RegisterJobResponse\x12J\n\x17ShareBatchAccessPattern\x12\x17.BatchAccessPatternList\x1a\x16.google.protobuf.Emptyb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'superdl.syncgrpc.protos.cache_coordinator_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_REGISTERJOBRESPONSE']._serialized_start=80
  _globals['_REGISTERJOBRESPONSE']._serialized_end=142
  _globals['_REGISTERJOBINFO']._serialized_start=144
  _globals['_REGISTERJOBINFO']._serialized_end=243
  _globals['_BATCHACCESSPATTERNLIST']._serialized_start=245
  _globals['_BATCHACCESSPATTERNLIST']._serialized_end=330
  _globals['_BATCH']._serialized_start=332
  _globals['_BATCH']._serialized_end=381
  _globals['_MESSAGE']._serialized_start=383
  _globals['_MESSAGE']._serialized_end=409
  _globals['_CACHECOORDINATORSERVICE']._serialized_start=412
  _globals['_CACHECOORDINATORSERVICE']._serialized_end=568
# @@protoc_insertion_point(module_scope)
