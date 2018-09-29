import tensorflow as tf 

class resnet:
    x_training = tf.placeholder(tf.float32,[None,156600])
    x_testing = tf.placeholder(tf.float32,[None,156600])
    batch_size = 64

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block="identity_block", training="training"):
        block_name = "res_"+ block +"_"+stage
        f1,f2,f3 = out_filters
        with tf.name_scope(block_name):
            X_shortcut = X_input

            conv_1 = tf.Variable(tf.random_normal([1,1,in_filter,f1]))
            X = tf.nn.conv2d(X_input,conv_1,[1,1,1,1],"SAME")
            X = tf.layers.batch_normalization(X,3)
            X = tf.nn.relu(X)

            conv_2 = tf.Variable(tf.random_normal([kernel_size,kernel_size,f1,f2]))
            X_1 = tf.nn.conv2d(X,conv_2,[1,1,1,1],"SAME")
            X_1 = tf.layers.batch_normalization(X_1,3)
            X_1 = tf.nn.relu(X_1)

            conv_3 = tf.Variable(tf.random_normal([kernel_size,kernel_size,f2,f3]))
            X_2 = tf.nn.conv2d(X_1,conv_3,[1,1,1,1],"SAME")
            X_2 = tf.layers.batch_normalization(X_2,3)

            X_result = tf.add(X_2,X_shortcut)
            X_result = tf.nn.relu(X_result)

        return X_result

    def convolutional_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training,stride = 2):
        block_name = "res_"+ block +"_"+stage
        f1,f2,f3 = out_filters
        with tf.name_scope(block_name):
            X_shortcut = X_input

            conv_1 = tf.Variable(tf.random_normal([1,1,in_filter,f1]))
            X = tf.nn.conv2d(X_input,conv_1,[1,stride,stride,1],"SAME")
            X = tf.layers.batch_normalization(X,3)
            X = tf.nn.relu(X)

            conv_2 = tf.Variable(tf.random_normal([kernel_size,kernel_size,f1,f2]))
            X_1 = tf.nn.conv2d(X,conv_2,[1,1,1,1],"SAME")
            X_1 = tf.layers.batch_normalization(X_1,3)
            X_1 = tf.nn.relu(X_1)

            conv_3 = tf.Variable(tf.random_normal([kernel_size,kernel_size,f2,f3]))
            X_2 = tf.nn.conv2d(X_1,conv_3,[1,1,1,1],"SAME")
            X_2 = tf.layers.batch_normalization(X_2,3)

            conv_shcut = tf.Variable(tf.random_normal([1,1,in_filter,f3]))
            X_SC = tf.nn.conv2d(X_shortcut,conv_shcut,[1,stride,stride,1],"SAME")
            X_SC = tf.layers.batch_normalization(X_SC,3)

            X_result = tf.add(X_2,X_SC)
            X_result = tf.nn.relu(X_result)

        return X_result

    def build_main_structure(self, x_input, classes=6):

        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3 ], [3, 3], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference') :
            training = tf.placeholder(tf.bool, name='training')

            #stage 1
            w_conv1 = tf.Variable(tf.random_normal([7, 7, 3, 64]))
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='VALID')

            #stage 2
            x = self.convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
            x = self.identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            #stage 3
            x = self.convolutional_block(x, 3, 256, [128,128,512], 3, 'a', training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'b', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'c', training=training)
            x = self.identity_block(x, 3, 512, [128,128,512], 3, 'd', training=training)

            #stage 4
            x = self.convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = self.identity_block (x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x = self.identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            return x           
