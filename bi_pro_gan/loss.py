#Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def hinge(x):
    return tf.nn.relu( 1.0 - x )

#----------------------------------------------------------------------------
# Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size,
    cond_weight = 1.0): # Weight of the conditioning term.

    #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents = tf.random_uniform( [minibatch_size] + G.input_shapes[0][1:], -1.0, 1.0)
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,latents,is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,is_training=True))
    loss = -fake_scores_out

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, latents=None,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents = tf.random_uniform( [minibatch_size] + G.input_shapes[0][1:], -1.0, 1.0)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #real_scores_out, real_labels_out = fp32(D.get_output_for(reals,latents, is_training=True))
    #fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,latents, is_training=True))
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals,is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        #mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out,latents, is_training=True))
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------

def G_lo_gan(G, D, E, opt, training_set, minibatch_size, reals, labels,
    cond_weight = 1.0): # Weight of the conditioning term.

    Z = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    X = reals
    labels = training_set.get_random_labels_tf(minibatch_size)
    
    G = G.get_output_for(Z, labels, is_training=True)
    L = E.get_output_for(X, is_training=True)
    
    D_o, _ = fp32( D.get_output_for(G,Z,is_training=True) )
    Do, _ = fp32( D.get_output_for(X,L,is_training=True) )
    
    loss = tf.reduce_mean(Do - D_o)

#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#         loss += label_penalty_fakes * cond_weight
    return loss

def E_lo_gan(G, D, E, opt, training_set, minibatch_size, reals, labels,
    cond_weight = 1.0): # Weight of the conditioning term.

    Z = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    X = reals
    labels = training_set.get_random_labels_tf(minibatch_size)
    
    G = G.get_output_for(Z, labels, is_training=True)
    L = E.get_output_for(X, is_training=True)
    
    D_o, _ = fp32( D.get_output_for(G,Z,is_training=True) )
    Do, _ = fp32( D.get_output_for(X,L,is_training=True) )
    
    loss = tf.reduce_mean(Do - D_o)

#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#         loss += label_penalty_fakes * cond_weight
    return loss

def D_lo_gan(G, D, E , opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.
    
    Z = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    X = reals
    labels = training_set.get_random_labels_tf(minibatch_size)
    
    G_out = G.get_output_for(Z, labels, is_training=True)
    L = E.get_output_for(X, is_training=True)
    
    D_o, _ = fp32( D.get_output_for(G_out,Z,is_training=True) )
    Do, _ = fp32( D.get_output_for(X,L,is_training=True) )
    
    X_hat = G.get_output_for(L, labels, is_training=True) 
    
    loss = tf.square(tf.nn.sigmoid(D_o)) + tf.square(1 - tf.nn.sigmoid(Do))
    loss = tf.reduce_mean(loss)
    
    fake_images_out = X_hat
    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out,L, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    
    real_scores_out = tfutil.autosummary('Loss/real_scores', Do)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', D_o)
    
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

#     if D.output_shapes[1][1] > 0:
#         with tf.name_scope('LabelPenalty'):
#             label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
#             label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
#             label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
#             label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
#         loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
        
    return loss
    

                                                    #BI-GAN LOSSES        
#---------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Discriminator loss function for training Bi-GAN

def gp_loss( mixed_scores, mixed_images, wgan_target, opt):
    mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores))
    mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images])[0]))
    mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
    mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
    gradient_penalty = tf.square(mixed_norms - wgan_target)
    return gradient_penalty

def D_bigan_acgan(G, D, E, opt, training_set, minibatch_size, reals, labels,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents = tf.random_uniform( [minibatch_size] + G.input_shapes[0][1:], -1.0, 1.0)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    
    L = E.get_output_for(reals, is_training=True)
    X_hat = G.get_output_for(L, labels, is_training=True)
    
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals,L, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,latents, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out
    
    #real_s_x, real_s_x_z = fp32(D.get_output_for(reals,L, is_training=True))
    #fake_s_x, fake_s_x_z = fp32(D.get_output_for(fake_images_out,latents, is_training=True))
    #loss = - real_s_x - real_s_x_z + fake_s_x + fake_s_x_z

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=X_hat.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, X_hat.dtype), X_hat, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out,L, is_training=True))
        #mixed_scores_s_x, mixed_scores_s_x_z = fp32(D.get_output_for(mixed_images_out,L, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
        #gradient_penalty = gp_loss( mixed_scores_s_x, mixed_images_out, wgan_target, opt)
        #gradient_penalty += gp_loss( mixed_scores_s_x_z, mixed_images_out, wgan_target, opt)
        
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
        #epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty_x', tf.square(real_s_x))
        #epsilon_penalty += tfutil.autosummary('Loss/epsilon_penalty_x_z', tf.square(real_s_x_z))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------  

#----------------------------------------------------------------------------
# Generator loss function used in the paper 

def G_bigan_acgan(G, D,E,  opt, training_set, minibatch_size, reals,
    cond_weight = 1.0): # Weight of the conditioning term.

    #latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    latents = tf.random_uniform( [minibatch_size] + G.input_shapes[0][1:], -1.0, 1.0)
    labels = training_set.get_random_labels_tf(minibatch_size)

    fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,latents,is_training=True))
    loss = -fake_scores_out
    
    #fake_s_x, fake_s_x_z = fp32(D.get_output_for(fake_images_out,latents,is_training=True))
    #loss = -fake_s_x - fake_s_x_z
    
    #New Addition
    #L = E.get_output_for(reals, is_training=True)
    #real_scores_out, real_labels_out = fp32(D.get_output_for(reals,L, is_training=True))
    #loss += real_scores_out

    #reconstruction Img loss
    #recon_imgs = G.get_output_for(L, labels, is_training=True)
    #ax1 = [i for i in range(1, len(reals.get_shape()))]
    #recon_imgs_loss = tf.reduce_mean( tf.abs(reals-recon_imgs), axis = ax1 )
    #loss += recon_imgs_loss
    #recon_imgs_loss = tf.reduce_mean( recon_imgs_loss )
    
    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight

    return loss

#----------------------------------------------------------------------------
# Encode loss function used in the paper 

def E_bigan_acgan(G, D, E, opt, training_set, minibatch_size,reals, labels,
    cond_weight = 1.0, eps = 0.001): # Weight of the conditioning term.

    latents = tf.random_uniform([minibatch_size] + G.input_shapes[0][1:], -1.0, 1.0)
    labels = training_set.get_random_labels_tf(minibatch_size)

    #reconstruction Z loss
    #fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #recon_latents = E.get_output_for(fake_images_out, is_training=True)
    #ax2 = [i for i in range(1, len(latents.get_shape()))]
    #recons_Z_loss = tf.reduce_mean( tf.abs(recon_latents-latents), axis = ax2  )

    #original encoder loss
    L = E.get_output_for(reals, is_training=True)
    real_scores_out, real_labels_out = fp32(D.get_output_for(reals,L, is_training=True))
    loss = real_scores_out
    
    #real_s_x, real_s_x_z = fp32(D.get_output_for(reals,L, is_training=True))
    #loss = real_s_x + real_s_x_z

    #New Addition
    #fake_images_out = G.get_output_for(latents, labels, is_training=True)
    #fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out,latents,is_training=True))
    #loss += -fake_scores_out

    #reconstruction Img loss
    #recon_imgs = G.get_output_for(L, labels, is_training=True)
    #recon_imgs_loss = tf.reduce_sum( tf.square(reals-recon_imgs), 0 )
    #recon_imgs_loss = tf.reduce_mean( recon_imgs_loss )

    #loss += 0.5 * recons_Z_loss #+ recon_imgs_loss

    return loss
