# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from superdl.protos import cache_coordinator_pb2 as super__dl_dot_protos_dot_cache__coordinator__pb2


class CacheCoordinatorServiceStub(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./super_dl/protos/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.RegisterJob = channel.unary_unary(
                '/CacheCoordinatorService/RegisterJob',
                request_serializer=super__dl_dot_protos_dot_cache__coordinator__pb2.JobInfo.SerializeToString,
                response_deserializer=super__dl_dot_protos_dot_cache__coordinator__pb2.RegisterJobResponse.FromString,
                )
        self.SendMetrics = channel.unary_unary(
                '/CacheCoordinatorService/SendMetrics',
                request_serializer=super__dl_dot_protos_dot_cache__coordinator__pb2.MetricsRequest.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.SendBatchAccessPattern = channel.unary_unary(
                '/CacheCoordinatorService/SendBatchAccessPattern',
                request_serializer=super__dl_dot_protos_dot_cache__coordinator__pb2.BatchAccessPatternList.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class CacheCoordinatorServiceServicer(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./super_dl/protos/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    def RegisterJob(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendMetrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendBatchAccessPattern(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CacheCoordinatorServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'RegisterJob': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterJob,
                    request_deserializer=super__dl_dot_protos_dot_cache__coordinator__pb2.JobInfo.FromString,
                    response_serializer=super__dl_dot_protos_dot_cache__coordinator__pb2.RegisterJobResponse.SerializeToString,
            ),
            'SendMetrics': grpc.unary_unary_rpc_method_handler(
                    servicer.SendMetrics,
                    request_deserializer=super__dl_dot_protos_dot_cache__coordinator__pb2.MetricsRequest.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'SendBatchAccessPattern': grpc.unary_unary_rpc_method_handler(
                    servicer.SendBatchAccessPattern,
                    request_deserializer=super__dl_dot_protos_dot_cache__coordinator__pb2.BatchAccessPatternList.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'CacheCoordinatorService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class CacheCoordinatorService(object):
    """
    Command to create stub files:
    python -m grpc_tools.protoc --proto_path=. ./super_dl/protos/cache_coordinator.proto --python_out=. --grpc_python_out=.
    """

    @staticmethod
    def RegisterJob(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/RegisterJob',
            super__dl_dot_protos_dot_cache__coordinator__pb2.JobInfo.SerializeToString,
            super__dl_dot_protos_dot_cache__coordinator__pb2.RegisterJobResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendMetrics(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/SendMetrics',
            super__dl_dot_protos_dot_cache__coordinator__pb2.MetricsRequest.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendBatchAccessPattern(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/CacheCoordinatorService/SendBatchAccessPattern',
            super__dl_dot_protos_dot_cache__coordinator__pb2.BatchAccessPatternList.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
