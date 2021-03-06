import tensorflow as tf
import random
#example需要serialize
#TFRecordWriter并不会自建文件夹。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
num_shards=2
instances_per_shards=2
for i in range(num_shards):
    filename=('./path/data.tfrecords-%.5d-of-%.5d'%(i,num_shards))
    writer=tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shards):
        # t=random.randint(0,10)
        # print(t)
        example=tf.train.Example(features=tf.train.Features(
            feature={
                'i':_int64_feature(i),
                'j':_int64_feature(j)
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()
