#!/bin/bash

# Datasets:
# Cats and dogs
files_to_download[0]=https://storage.googleapis.com/mlep-public/course_1/week2/kagglecatsanddogs_3367a.zip
# Caltech birds
files_to_download[1]=https://storage.googleapis.com/mlep-public/course_1/week2/CUB_200_2011.tar

# # Pretrained models and training histories
model_balanced_data[0]=https://storage.googleapis.com/mlep-public/course_1/week2/model-balanced/saved_model.pb
model_balanced_data[1]=https://storage.googleapis.com/mlep-public/course_1/week2/model-balanced/variables/variables.data-00000-of-00001
model_balanced_data[2]=https://storage.googleapis.com/mlep-public/course_1/week2/model-balanced/variables/variables.index
model_balanced_data[3]=https://storage.googleapis.com/mlep-public/course_1/week2/history-balanced/history-balanced.csv

model_imbalanced_data[0]=https://storage.googleapis.com/mlep-public/course_1/week2/model-imbalanced/saved_model.pb
model_imbalanced_data[1]=https://storage.googleapis.com/mlep-public/course_1/week2/model-imbalanced/variables/variables.data-00000-of-00001
model_imbalanced_data[2]=https://storage.googleapis.com/mlep-public/course_1/week2/model-imbalanced/variables/variables.index
model_imbalanced_data[3]=https://storage.googleapis.com/mlep-public/course_1/week2/history-imbalanced/history-imbalanced.csv

model_augmented_data[0]=https://storage.googleapis.com/mlep-public/course_1/week2/model-augmented/saved_model.pb
model_augmented_data[1]=https://storage.googleapis.com/mlep-public/course_1/week2/model-augmented/variables/variables.data-00000-of-00001
model_augmented_data[2]=https://storage.googleapis.com/mlep-public/course_1/week2/model-augmented/variables/variables.index
model_augmented_data[3]=https://storage.googleapis.com/mlep-public/course_1/week2/history-augmented/history-augmented.csv

download_data () {
arr=("$@")    
for file in "${arr[@]}"
do
    b_name="$(basename -- $file)"
    if [ -f "${1}/${b_name}" ]; then
    echo "File ${b_name} exists"
    else 
        echo "File ${b_name} does not exist"
        wget --directory-prefix=$1 $file
    fi 
done
}

download_data ./archives ${files_to_download[*]}  
download_data ./archives/models_balanced ${model_balanced_data[*]}
download_data ./archives/models_imbalanced ${model_imbalanced_data[*]} 
download_data ./archives/models_augmented ${model_augmented_data[*]} 