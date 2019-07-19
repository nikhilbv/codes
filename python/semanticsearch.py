python demo.py --features_path feat_4096 --file_mapping_path index_4096 --model_path my_model.hdf5 --custom_features_path feat_300 --custom_features_file_mapping_path index_300 --search_key 872 --train_model True --generate_image_features True --generate_custom_features True --training_epochs 1 --glove_model_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B --data_path dataset


/home/prime/Documents/ai-ml-dl-data/data/glove.6B


python search.py --index_folder dataset --features_path feat_4096 --file_mapping index_4096 --index_boolean True --features_from_new_model_boolean False



python search.py --input_image dataset/cat/2008_001335.jpg --features_path feat_4096 --file_mapping index_4096 --index_boolean False --features_from_new_model_boolean False
python search.py --input_word cat --features_path feat_4096 --file_mapping index_4096 --index_boolean False --features_from_new_model_boolean False



[[581, 'dataset/cat/2008_001335.jpg', 0.0]
, [579, 'dataset/cat/2008_001290.jpg', 0.9706233739852905]
, [590, 'dataset/cat/2008_002294.jpg', 0.9859563708305359]
, [563, 'dataset/cat/2008_003622.jpg', 0.9882920980453491]
, [597, 'dataset/cat/2008_004347.jpg', 0.994615375995636]
, [566, 'dataset/cat/2008_005469.jpg', 1.0203195810317993]
, [580, 'dataset/cat/2008_005386.jpg', 1.028808355331421]
, [556, 'dataset/cat/2008_007496.jpg', 1.0298227071762085]
, [569, 'dataset/cat/2008_002201.jpg', 1.0411686897277832]
, [582, 'dataset/cat/2008_000345.jpg', 1.0663559436798096]]


python train.py --model_save_path my_model.hdf5 --checkpoint_path checkpoint.hdf5 --glove_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B --dataset_path dataset --num_epochs 2  


python search.py --index_folder dataset --features_path feat_300 --file_mapping index_300 --model_path my_model.hdf5 --index_boolean True --features_from_new_model_boolean True --glove_path models/glove.6B
python search.py --index_folder dataset --features_path feat_300 --file_mapping index_300 --model_path my_model.hdf5 --index_boolean True --features_from_new_model_boolean True --glove_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B


/home/prime/Documents/ai-ml-dl/external/semantic-search/my_model.hdf5


python search.py --input_image dataset/cat/2008_001335.jpg --features_path feat_300 --file_mapping index_300 --model_path my_model.hdf5 --index_boolean False --features_from_new_model_boolean True --glove_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B


python search.py --input_word cat --features_path feat_300 --file_mapping index_300 --model_path my_model.hdf5 --index_boolean False --features_from_new_model_boolean True --glove_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B


python demo.py --features_path feat_4096 --file_mapping_path index_4096 --model_path my_model.hdf5 --custom_features_path feat_300 --custom_features_file_mapping_path index_300 --search_key 872 --train_model False --generate_image_features False --generate_custom_features False --glove_path /home/prime/Documents/ai-ml-dl-data/data/glove.6B

python main.py train --arch cfg/arch/mask_rcnn-260419_163410.yml --dataset cfg/dataset/260419_163410.yml > /home/prime/Documents/ai-ml-dl-data/logs/mask_rcnn/train_hmd_260419.2.log &

/home/prime/Documents/ai-ml-dl-data/logs/mask_rcnn/train_hmd_290419.1.log

python main.py train --arch cfg/arch/mask_rcnn-260419_143621.yml --dataset cfg/dataset/260419_143621.yml > /home/prime/Documents/ai-ml-dl-data/logs/mask_rcnn/train_hmd_290419.1.log &
