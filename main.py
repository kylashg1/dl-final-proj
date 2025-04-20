import tensorflow as tf

def main():
    print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

if __name__ ==  "__main__":
    main()
