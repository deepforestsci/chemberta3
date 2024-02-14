from ray_ds import RayDataset

if __name__ == '__main__':
    dataset_path = 's3://chemberta3/ray_test/featurized_data/MolGraphConv/zinc250k/'
    train_dataset = RayDataset.read(dataset_path).dataset
    for batch in train_dataset.iter_batches():
        print(batch)
        break
