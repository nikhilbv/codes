import tensorflow as tf

## Creates a graph.
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))


# >>> c.
# c.ByteSize(                     c.DiscardUnknownFields(         c.IsInitialized(                c.SerializeToString(            c.device_filters                c.log_device_placement
# c.Clear(                        c.Experimental(                 c.ListFields(                   c.SetInParent(                  c.experimental                  c.operation_timeout_in_ms
# c.ClearExtension(               c.Extensions                    c.MergeFrom(                    c.UnknownFields(                c.gpu_options                   c.placement_period
# c.ClearField(                   c.FindInitializationErrors(     c.MergeFromString(              c.WhichOneof(                   c.graph_options                 c.rpc_options
# c.CopyFrom(                     c.FromString(                   c.ParseFromString(              c.allow_soft_placement          c.inter_op_parallelism_threads  c.session_inter_op_thread_pool
# c.DESCRIPTOR                    c.HasExtension(                 c.RegisterExtension(            c.cluster_def                   c.intra_op_parallelism_threads  c.use_per_session_threads
# c.DeviceCountEntry(             c.HasField(                     c.SerializePartialToString(     c.device_count                  c.isolate_session_state         
# >>> c.gpu_options
# c.gpu_options
# >>> c.gpu_options.
# c.gpu_options.ByteSize(                        c.gpu_options.FindInitializationErrors(        c.gpu_options.RegisterExtension(               c.gpu_options.experimental
# c.gpu_options.Clear(                           c.gpu_options.FromString(                      c.gpu_options.SerializePartialToString(        c.gpu_options.force_gpu_compatible
# c.gpu_options.ClearExtension(                  c.gpu_options.HasExtension(                    c.gpu_options.SerializeToString(               c.gpu_options.per_process_gpu_memory_fraction
# c.gpu_options.ClearField(                      c.gpu_options.HasField(                        c.gpu_options.SetInParent(                     c.gpu_options.polling_active_delay_usecs
# c.gpu_options.CopyFrom(                        c.gpu_options.IsInitialized(                   c.gpu_options.UnknownFields(                   c.gpu_options.polling_inactive_delay_msecs
# c.gpu_options.DESCRIPTOR                       c.gpu_options.ListFields(                      c.gpu_options.WhichOneof(                      c.gpu_options.visible_device_list
# c.gpu_options.DiscardUnknownFields(            c.gpu_options.MergeFrom(                       c.gpu_options.allocator_type                   
# c.gpu_options.Experimental(                    c.gpu_options.MergeFromString(                 c.gpu_options.allow_growth                     
# c.gpu_options.Extensions                       c.gpu_options.ParseFromString(                 c.gpu_options.deferred_deletion_bytes    