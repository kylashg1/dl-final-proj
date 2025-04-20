import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

if __name__ ==  "__main__":
    main()
