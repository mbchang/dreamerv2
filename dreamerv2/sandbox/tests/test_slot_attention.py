import tensorflow as tf

import slot_attention_utils as utils
import slot_attention_learners as model_utils

def test_slot_attention():
    learning_rate = 0.001
    resolution = (128, 128)
    # resolution = (64, 64)
    batch_size = 2
    num_slots = 3
    num_iterations = 2
    num_steps = 5

    tf.random.set_seed(0)

    optimizer = tf.keras.optimizers.Adam(learning_rate, epsilon=1e-08)

    model = model_utils.build_model(
        resolution, batch_size, num_slots, num_iterations,
        model_type="object_discovery")

    input_shape = (batch_size, resolution[0], resolution[1], 3)
    random_input = tf.random.uniform(input_shape)

    for i in range(num_steps):
        with tf.GradientTape() as tape:
          preds = model(random_input, training=True)
          recon_combined, _, _, _ = preds
          loss_value = utils.l2_loss(random_input, recon_combined)

        # Get and apply gradients.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        print(f'Iter: {i} Loss: {loss_value}')


if __name__ == '__main__':
    test_slot_attention()
