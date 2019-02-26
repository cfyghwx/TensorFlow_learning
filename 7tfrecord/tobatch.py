import tensorflow as tf
#lobal_variables_initializer()在新版本中将代替initialize_all_variables()
#在本例中需要使用local_variables_initializer()
#
files=tf.train.match_filenames_once("./path/data.tfrecords-*")
filename_queue=tf.train.string_input_producer(files,num_epochs=10,shuffle=False)
# filename_queue=tf.data.Dataset.from_tensor_slices(files).repeat(4)
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(serialized_example,
                                 features={
                                     'i':tf.FixedLenFeature([],tf.int64),
                                     'j':tf.FixedLenFeature([],tf.int64)

                                 })
example,label=features['i'],features['j']
batch_size=3
capacity=1000+3*batch_size

example_batch,label_batch=tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    # tf.local_variables_initializer().run()
    # tf.global_variables_initializer().run()
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    #
    for i in range(2):
        cur_example_batch,cur_label_batch=sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    # for i in range(2):
    #     print(sess.run([example,label]))
    coord.request_stop()
    coord.join(threads)