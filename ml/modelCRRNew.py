from __future__ import print_function
import os
import time
import random
import tensorflow as tf
import numpy as np
import cv2
from ml.utils import *
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model


def concat(layers):
    return tf.concat(layers, axis=3)

def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_lum = input_im[:, :, :, 0:1]
    input_chroma = input_im[:, :, :, 1:3]
    
    # Define all layers first (this makes them trainable)
    initial_conv = tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu)
    residual_convs = [tf.keras.layers.Conv2D(channel, kernel_size, padding='same', activation=tf.nn.relu) 
                     for _ in range(layer_num)]
    channel_dense1 = tf.keras.layers.Dense(channel//4, activation='relu')
    channel_dense2 = tf.keras.layers.Dense(channel, activation='sigmoid')
    spatial_convs = [tf.keras.layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid') 
                    for _ in range(layer_num)]
    strength_conv = tf.keras.layers.Conv2D(2, kernel_size, padding='same', activation='sigmoid')
    offset_conv = tf.keras.layers.Conv2D(2, kernel_size, padding='same')

    # Build the computation graph
    x = initial_conv(input_chroma)
    
    channel_attentions = []
    spatial_attentions = []
    
    for i in range(layer_num):
        # Channel attention path
        residual = residual_convs[i](x)
        squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)
        excitation = channel_dense1(squeeze)
        excitation = channel_dense2(excitation)
        channel_attention = excitation
        residual = residual * channel_attention
        
        # Spatial attention path
        spatial = tf.reduce_mean(residual, axis=3, keepdims=True)
        spatial_attention = spatial_convs[i](spatial)
        residual = residual * spatial_attention
        
        x += residual
        x = tf.nn.relu(x)
        
        channel_attentions.append(channel_attention)
        spatial_attentions.append(spatial_attention)
    
    # Final processing
    final_channel_attention = channel_attentions[-1]
    final_spatial_attention = spatial_attentions[-1]
    
    strength_mask = strength_conv(x)
    chroma_offset = offset_conv(x)
    corrected_chroma = input_chroma - (chroma_offset * strength_mask)
    
    # Luminance remains unchanged
    lum_enhanced = input_lum

    # Final outputs
    R = tf.concat([lum_enhanced, corrected_chroma], axis=-1)
    I = tf.ones_like(input_lum)
    
    return R, R, I, final_channel_attention, final_spatial_attention

def color_balance_target_loss(original_ab, corrected_ab):
    """Loss that encourages a 0.5 blend between original and neutral colors"""
    neutral_ab = tf.ones_like(original_ab) * 0.5
    target_ab = original_ab * 0.5 + neutral_ab * 0.5
    
    return tf.reduce_mean(tf.abs(corrected_ab - target_ab))

class ColorCastRemoval(tf.keras.Model):
    def __init__(self):
        super(ColorCastRemoval, self).__init__()
        self.DecomNet_layer_num = 5
        
        # Initialize all DecomNet layers here
        self.initial_conv = tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu)
        self.residual_convs = [tf.keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu) 
                             for _ in range(self.DecomNet_layer_num)]
        self.channel_dense1 = tf.keras.layers.Dense(16, activation='relu')  # 64//4=16
        self.channel_dense2 = tf.keras.layers.Dense(64, activation='sigmoid')
        self.spatial_convs = [tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')
                            for _ in range(self.DecomNet_layer_num)]
        self.strength_conv = tf.keras.layers.Conv2D(2, 3, padding='same', activation='sigmoid')
        self.offset_conv = tf.keras.layers.Conv2D(2, 3, padding='same')
        
        # VGG for perceptual loss
        self.vgg19 = VGG19(include_top=False, weights='imagenet')
        self.vgg19.trainable = False
        self.perceptual_loss_layer = Model(
            inputs=self.vgg19.input, 
            outputs=self.vgg19.get_layer('block4_conv2').output
        )
        
        print("[*] Enhanced ColorCastRemoval model initialized (LAB input)")
        print("[*] Trainable variables:", len(self.trainable_variables))
    
    def call(self, inputs, training=False):
        input_lum = inputs[:, :, :, 0:1]
        input_chroma = inputs[:, :, :, 1:3]
        
        # Build computation graph using registered layers
        x = self.initial_conv(input_chroma)
        
        channel_attentions = []
        spatial_attentions = []
        
        for i in range(self.DecomNet_layer_num):
            # Channel attention path
            residual = self.residual_convs[i](x)
            squeeze = tf.reduce_mean(residual, axis=[1,2], keepdims=True)
            excitation = self.channel_dense1(squeeze)
            excitation = self.channel_dense2(excitation)
            channel_attention = excitation
            residual = residual * channel_attention
            
            # Spatial attention path
            spatial = tf.reduce_mean(residual, axis=3, keepdims=True)
            spatial_attention = self.spatial_convs[i](spatial)
            residual = residual * spatial_attention
            
            x += residual
            x = tf.nn.relu(x)
            
            channel_attentions.append(channel_attention)
            spatial_attentions.append(spatial_attention)
        
        # Final processing for chrominance correction
        strength_mask = self.strength_conv(x)
        chroma_offset = self.offset_conv(x)
        corrected_chroma = input_chroma - (chroma_offset * strength_mask)
        
        # Estimate illumination map
        illumination_map = self.estimate_illumination(x)
        
        # Normalize illumination map to [0, 1]
        illumination_map = tf.nn.sigmoid(illumination_map)
        
        # Create outputs - THIS IS THE FIXED PART
        output_R = tf.concat([input_lum, corrected_chroma], axis=-1)
        
        # Enhanced image should use the corrected chroma, not apply illumination to it
        # Rather than blending, we should just use the corrected image as the enhanced version
        output_R_corrected = output_R  # Use this directly, don't apply illumination map again
        
        output_I = illumination_map
        final_channel_attention = channel_attentions[-1]
        final_spatial_attention = spatial_attentions[-1]
        
        # Ensure outputs are within [0, 1] range
        output_R_corrected = tf.clip_by_value(output_R_corrected, 0, 1)
        output_I = tf.clip_by_value(output_I, 0, 1)
        
        # Store as attributes for later access if needed
        self.output_R = output_R
        self.output_R_corrected = output_R_corrected
        self.output_I = output_I
        self.channel_attention = final_channel_attention
        self.spatial_attention = final_spatial_attention
        
        # Return all main outputs as a tuple
        return output_R, output_R_corrected, output_I, final_channel_attention, final_spatial_attention

    def estimate_illumination(self, x):
        """Estimate the illumination map from the feature map."""
        # Use a convolutional layer to estimate the illumination map
        illumination_conv = tf.keras.layers.Conv2D(1, 3, padding='same', activation='linear')
        illumination_map = illumination_conv(x)
        return illumination_map
    
    def build_model(self):
        # This method defines the core network architecture
        # No need to explicitly define as the forward pass will build it
        pass
    
    def calculate_histogram_loss(self, pred, target):
        """Modified histogram loss calculation that works with integer histograms"""
        # Scale and convert to integers for histogram calculation
        scale = 1000.0  # Scaling factor to maintain precision
        pred_scaled = tf.cast(pred * scale, tf.int32)
        target_scaled = tf.cast(target * scale, tf.int32)
        
        # Calculate histograms with integer type
        hist_pred = tf.histogram_fixed_width(pred_scaled, 
                                        [0, tf.cast(scale, tf.int32)], 
                                        nbins=50)
        hist_target = tf.histogram_fixed_width(target_scaled, 
                                            [0, tf.cast(scale, tf.int32)], 
                                            nbins=50)
        
        # Convert back to float for calculations
        hist_pred = tf.cast(hist_pred, tf.float32)
        hist_target = tf.cast(hist_target, tf.float32)
        
        # Normalize histograms
        hist_pred = hist_pred / tf.reduce_sum(hist_pred)
        hist_target = hist_target / tf.reduce_sum(hist_target)
        
        # Calculate EMD (Earth Mover's Distance)
        cum_pred = tf.cumsum(hist_pred)
        cum_target = tf.cumsum(hist_target)
        
        return tf.reduce_mean(tf.abs(cum_pred - cum_target))
        
    def calculate_color_balance(self, R):
        """
        Calculate color balance loss to ensure balanced channel representation.
        """
        r_mean = tf.reduce_mean(R[:, :, :, 0])
        g_mean = tf.reduce_mean(R[:, :, :, 1])
        b_mean = tf.reduce_mean(R[:, :, :, 2])
        
        balance_loss = tf.abs(r_mean - g_mean) + tf.abs(g_mean - b_mean) + tf.abs(b_mean - r_mean)
        return balance_loss

    def calculate_color_deviation(self, R):
        """
        Calculate color deviation loss to penalize extreme color biases.
        """
        r_mean = tf.reduce_mean(R[:, :, :, 0])
        g_mean = tf.reduce_mean(R[:, :, :, 1])
        b_mean = tf.reduce_mean(R[:, :, :, 2])
        
        overall_mean = (r_mean + g_mean + b_mean) / 3.0
        deviation = tf.abs(r_mean - overall_mean) + tf.abs(g_mean - overall_mean) + tf.abs(b_mean - overall_mean)
        return deviation

    def detail_preservation_loss(self, original, corrected):
        """Focus on chrominance details only (a/b channels)"""
        # Convert inputs to tensors
        original_tensor = tf.convert_to_tensor(original, dtype=tf.float32)
        corrected_tensor = tf.convert_to_tensor(corrected, dtype=tf.float32)
        
        # Extract only chrominance channels (a and b)
        orig_ab = original_tensor[..., 1:3]  # Shape: [batch, H, W, 2]
        corr_ab = corrected_tensor[..., 1:3]  # Shape: [batch, H, W, 2]
        
        # Calculate gradients only for chrominance
        orig_grad = tf.image.sobel_edges(orig_ab)
        corr_grad = tf.image.sobel_edges(corr_ab)
        
        return tf.reduce_mean(tf.abs(orig_grad - corr_grad))
    
    def calculate_perceptual_loss(self, output, target):
        """Enhanced perceptual loss using multiple VGG layers"""
        vgg_output = self.vgg19(preprocess_input(output * 255))
        vgg_target = self.vgg19(preprocess_input(target * 255))
        return tf.reduce_mean(tf.square(vgg_output - vgg_target))
    
    def compute_losses(self, input_low, input_high):
        # Convert inputs to tensors if they aren't already
        input_low = tf.convert_to_tensor(input_low, dtype=tf.float32)
        input_high = tf.convert_to_tensor(input_high, dtype=tf.float32)
        
        # Forward pass to get outputs
        enhanced = self(input_low)  # This calls the model's call method
        
        # Ensure we're using the corrected output (R_corrected)
        enhanced = self.output_R_corrected
        
        # Rest of your loss calculations...
        self.color_balance_loss = self.calculate_color_balance(self.output_R)
        self.color_deviation_loss = self.calculate_color_deviation(self.output_R)
        self.detail_loss = self.detail_preservation_loss(input_low, enhanced)
        self.quality_loss = 1.0 - tf.reduce_mean(tf.image.ssim(enhanced, input_high, 1.0))
        
        # Calculate perceptual loss
        self.perceptual_loss = self.calculate_perceptual_loss(enhanced, input_high)
        
        # New chromaticity loss components
        self.neutral_ab = tf.ones_like(enhanced[:,:,:,1:]) * 0.5
        self.chroma_loss = tf.reduce_mean(tf.abs(enhanced[:,:,:,1:] - self.neutral_ab))
        
        # Histogram matching loss
        self.hist_loss = self.calculate_histogram_loss(enhanced[:,:,:,1:], self.neutral_ab)
        
        self.recon_loss = tf.reduce_mean(tf.abs(enhanced - input_high))
        self.color_loss = tf.reduce_mean(tf.abs(enhanced[:,:,:,1:] - input_high[:,:,:,1:]))
        
        # Updated total loss
        self.total_loss = (1.0 * self.recon_loss + 
                         0.7 * self.chroma_loss +
                         0.5 * self.hist_loss +
                         0.3 * self.detail_loss)
        
        return self.total_loss

    def train(self, train_low_data, train_high_data, batch_size, epoch, lr_initial=0.001, ckpt_dir="./checkpoints"):
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        # Setup checkpoint manager
        checkpoint = tf.train.Checkpoint(model=self)
        manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=5)
        
        # Convert numpy arrays to TensorFlow Dataset for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices((train_low_data, train_high_data))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        # Training loop
        for ep in range(epoch):
            current_lr = lr_initial * (0.95 ** ep)
            optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
            
            epoch_loss = 0
            batch_num = 0
            
            # Iterate over batches using the dataset
            for batch_low, batch_high in train_dataset:
                with tf.GradientTape() as tape:
                    loss = self.compute_losses(batch_low, batch_high)
                
                # Get gradients
                print("Number of trainable variables:", len(self.trainable_variables))
                gradients = tape.gradient(loss, self.trainable_variables)
                print("Number of gradients:", len([g for g in gradients if g is not None]))
      
                # Apply gradients if they exist
                if gradients is not None:
                    optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                epoch_loss += loss.numpy()
                batch_num += 1
                
                print(f"Epoch: {ep+1} Batch: {batch_num} "
                    f"Loss: {loss.numpy():.4f} (recon={self.recon_loss.numpy():.4f}, "
                    f"chroma={self.chroma_loss.numpy():.4f}, "
                    f"hist={self.hist_loss.numpy():.4f}, detail={self.detail_loss.numpy():.4f})")
            
            # Save checkpoint and print stats
            manager.save()
            avg_loss = epoch_loss / batch_num
            print(f"Epoch {ep+1} Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
            print(f"[*] Model saved at epoch {ep+1} in {ckpt_dir}")
        
    def load(self, ckpt_dir):
        # Setup checkpoint for loading models
        checkpoint = tf.train.Checkpoint(model=self)
        
        # Try to restore the latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"[*] Loaded checkpoint from {latest_checkpoint}")
            return True
        print("[!] No valid checkpoint found.")
        return False

    def test(self, test_low_data, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for idx, img in enumerate(test_low_data):
            input_img = np.expand_dims(img, axis=0)
            
            # Process image
            enhanced = self(input_img)
            
            # Get specific outputs we need
            R_corrected = self.output_R_corrected
            I_map = self.output_I
            
            # Convert to numpy and clip values
            corrected = np.clip(R_corrected[0].numpy() * 255, 0, 255).astype(np.uint8)
            enhanced_img = np.clip(enhanced[0].numpy() * 255, 0, 255).astype(np.uint8)
            
            # Save outputs
            cv2.imwrite(os.path.join(save_dir, f'corrected_{idx}.png'), cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, f'enhanced_{idx}.png'), cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))
            print(f"Saved corrected image: {os.path.join(save_dir, f'corrected_{idx}.png')}")
            print(f"Saved enhanced image: {os.path.join(save_dir, f'enhanced_{idx}.png')}")