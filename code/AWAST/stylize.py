# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import vgg

import tensorflow as tf
import numpy as np

from sys import stderr

from PIL import Image

from pagerank import *

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

try:
    reduce
except NameError:
    from functools import reduce

def sigmoid(x):
  return float(1) / (1 + np.exp(-x))

def enforceWeight(arr):
    sigarr = sigmoid(-np.exp(1) + 2.0 * np.exp(1) * (arr - np.min(arr))/(np.max(arr) - np.min(arr)))
    # print(-np.exp(1) + 2.0 * np.exp(1) * (arr - np.min(arr))/(np.max(arr) - np.min(arr)))
    # print(sigarr)
    return sigarr / np.sum(sigarr)

def stylize(stylename, network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image); `iteration` is None
    if this is the final image (the last iteration).  Other tuples are yielded
    every `checkpoint_iterations` iterations.

    :rtype: iterator[tuple[int|None,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)


    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image = tf.placeholder('float32', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            image = tf.placeholder('float32', shape=style_shapes[i])

            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))

                a = tf.placeholder('float32', features.shape)
                b = tf.placeholder('float32', features.T.shape)
                c = tf.matmul(b, a)
                sess = tf.Session()

                gram = sess.run(c, feed_dict={a: features, b:features.T})
                # print(np.sum(gram))
                gram = gram / features.size
                # print(np.sum(gram))
                style_features[i][layer] = gram

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # automatic layer weight and blend weight

    #calculate style similarity in different layer

    style_diff = np.zeros(shape = (len(STYLE_LAYERS), len(styles), len(styles)))

    for l in range(len(STYLE_LAYERS)):
        for i in range(len(styles)):
            for j in range(i + 1,len(styles)):
                # print(np.sum(style_features[i][STYLE_LAYERS[l]]))
                # print(np.linalg.norm(style_features[i][STYLE_LAYERS[l]] - style_features[j][STYLE_LAYERS[l]], 2))
                style_diff[l, j, i] = np.linalg.norm(style_features[i][STYLE_LAYERS[l]] - style_features[j][STYLE_LAYERS[l]], 2)
                style_diff[l, i, j] = style_diff[l, j, i]
    # print(style_diff[1])
    # print(style_diff[2])
    # print(style_diff[1][1])
    weightPR = {}

    for l in range(len(STYLE_LAYERS)):
        style_diff[l] /= np.max(style_diff[l])
        # print(style_diff[l])
        res = np.array(pageRankUndirected(style_diff[l]))
        # print(style_diff[l])
        res = enforceWeight(res)
        weightPR[STYLE_LAYERS[l]] = res
        ind = np.arange(len(styles)).reshape(-1, 1)
        res = res.reshape(-1, 1)
        res = np.concatenate((ind, res), axis=1)
        # print (res)
        res = res[res[:,1].argsort()[::-1]]
        # print("Style weight in layer: " + STYLE_LAYERS[l])
        # for i in range(len(styles)):
        #     print("\tNo." + str(i + 1) + "\t" + str(res[i, 1]) + "\tof " + stylename[int(res[i, 0])])

    for l in STYLE_LAYERS:
        weightPR[l] /= float(len(STYLE_LAYERS))

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = tf.random_normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype('float32')
            noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
            initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (1.0 - initial_content_noise_coeff)
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        # content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(
                    net[content_layer] - content_features[content_layer]) /
                    content_features[content_layer].size))
        content_loss += reduce(tf.add, content_losses)

        # style loss #########################
        style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_losses.append(weightPR[style_layer][i] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
                # style_losses.append(style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            # style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
            style_loss += style_weight * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:,1:,:,:])
        tv_x_size = _tensor_size(image[:,:,1:,:])
        tv_loss = tv_weight * 2 * (
                (tf.nn.l2_loss(image[:,1:,:,:] - image[:,:shape[1]-1,:,:]) /
                    tv_y_size) +
                (tf.nn.l2_loss(image[:,:,1:,:] - image[:,:,:shape[2]-1,:]) /
                    tv_x_size))
        # overall loss
        loss = content_loss + style_loss + tv_loss

        # optimizer setup
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        def print_progress():
            stderr.write('  content loss: %g\n' % content_loss.eval())
            stderr.write('    style loss: %g\n' % style_loss.eval())
            stderr.write('       tv loss: %g\n' % tv_loss.eval())
            stderr.write('    total loss: %g\n' % loss.eval())

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            if (print_iterations and print_iterations != 0):
                print_progress()
            for i in range(iterations):
                stderr.write('Iteration %4d/%4d\n' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step or (print_iterations and i % print_iterations == 0):
                    print_progress()

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors and preserve_colors == True:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))


                    yield (
                        (None if last_step else i),
                        img_out
                    )


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb
