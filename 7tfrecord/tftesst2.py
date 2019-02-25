import tensorflow as tf
#最好写一下轮次，不然会错
files=tf.train.match_filenames_once("./path/data.tfrecords-*")
filename_queue=tf.train.string_input_producer(files,num_epochs=1,shuffle=False)
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features=tf.parse_single_example(serialized_example,
                                 features={
                                     'i':tf.FixedLenFeature([],tf.int64),
                                     'j':tf.FixedLenFeature([],tf.int64)

                                 })
with tf.Session() as sess:
    tf.local_variables_initializer().run()
    print(files)
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(2):
        print(sess.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)