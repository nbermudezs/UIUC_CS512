#!/bin/bash
DATA_FOLDER="data/"
DATA_FILES=(YELP DBLP)
RESULTS_FOLDER="../UIUC_CS512/assignment_1/results/"
AUTO_PHRASE_ROOT="../../AutoPhrase"
WORD2VEC_ROOT="../../word2vec/trunk"
SEGMENTATION_PARAMS_MULTI=(0.9 0.7 0.3 0.1 0.5)
SEGMENTATION_PARAMS_SINGLE=(0.1 0.3 0.5 0.7 0.9)
N_CLUSTERS=(10 20 40 80)

# activate environment
source activate py36

cd $AUTO_PHRASE_ROOT;
for file_type in ${DATA_FILES[@]}
    do
        model="models/"$file_type
        if [[ ! -f "$model/segmentation.model" ]]; then
            data=$DATA_FOLDER$file_type".txt"

            echo "Performing auto_phrase.sh MODEL=$model RAW_TRAIN=$data"
            export MODEL=$model RAW_TRAIN=$data
            bash auto_phrase.sh
        fi
    done
cd -

for file_type in ${DATA_FILES[@]}
    do
        cd $AUTO_PHRASE_ROOT;

        data=$DATA_FOLDER$file_type".txt"
        model="models/"$file_type

        for multi in ${SEGMENTATION_PARAMS_MULTI[@]}
            do
                for single in ${SEGMENTATION_PARAMS_SINGLE[@]}
                do
                    if [[ -f "$model/segmentation-m"$multi"s"$single".txt" ]]; then
                        continue
                    fi

                    if [[ -f "$model/segmentation.txt" ]]; then
                        rm $model"/segmentation.txt"
                    fi

                    echo "Performing phrasal_segmentation.sh \
                        MODEL=$model \
                        TEXT_TO_SEG=$data \
                        HIGHLIGHT_MULTI=$multi \
                        HIGHLIGHT_SINGLE=$single"

                    export MODEL=$model
                    export TEXT_TO_SEG=$data
                    export HIGHLIGHT_MULTI=$multi
                    export HIGHLIGHT_SINGLE=$single
                    bash phrasal_segmentation.sh | tee - a $model"/tmp-m"$multi"s"$single".txt"
                    mv $model"/tmp-m"$multi"s"$single".txt" \
                        $model"/segmentation-m"$multi"s"$single".txt"

                    # if we did care about the segmentation file we would do
                    # this instead.
                    # cp $model"/segmentation.txt" \
                    #    $model"/segmentation-m"$multi"s"$single".txt"
                done
            done

        # replace the <phrase> tags with __ and the spaces inside to _ to trick
        # word2vec into thinking the phrase is a single word
        if [[ ! -f $model"/segmentation-replaced.txt" ]]; then
            echo "Replacing <phrase> in segmented file"
            perl -pe's/<phrase>(.+?)<\/phrase>/"__".$1=~s| |_|gr."__"/ge' \
                $model"/segmentation.txt" > $model"/segmentation-replaced.txt"
        fi

        if [[ ! -f $model"/phrase.emb" ]]; then
            cd -
            echo "Running word2vec on last segmented file: "$AUTO_PHRASE_ROOT"/"$model"/segmentation-replaced.txt"
            cd $WORD2VEC_ROOT

            ./word2vec -train $AUTO_PHRASE_ROOT"/"$model"/segmentation-replaced.txt" \
                -output $AUTO_PHRASE_ROOT"/"$model"/phrase.emb" -cbow 0 -size 100 -window 5 \
                -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 0

            cd -
        fi


        if [[ ! -f $model"/phrase-filtered.emb" ]]; then
            cd $AUTO_PHRASE_ROOT
            cat $model"/phrase.emb" | grep "__\(.*\)__" \
                > $model"/phrase-filtered.emb"
            cd -
        fi


        for n_cluster in ${N_CLUSTERS[@]}
        do
            echo "Performing clustering. n_clusters="$n_cluster
            cluster_folder="clusters-"$n_cluster
            python main.py --task=clustering \
                --data_folder=$AUTO_PHRASE_ROOT"/"$model \
                --filename=phrase-filtered.emb \
                --cluster_folder=$cluster_folder \
                --n_clusters=$n_cluster
        done

        echo "Done with $file_type"
        echo "----------------------------------------------------------------"
    done

