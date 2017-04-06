datadir="../RNNLM_Penn/"

#use allData.txt to generate vocabulary list
cat $datadir/allData.txt | awk '{for (i=1;i<=NF;i++){print $i}}' | sort | uniq -c | sort -nrk1 > $datadir/counts.txt
echo -e '(eee)' > $datadir/voc.txt
awk '{print $2}' $datadir/counts.txt >> $datadir/voc.txt

#map word into integer for vocabulary
awk 'BEGIN{i=1}{print $0"(-)"i}{i=i+1}' $datadir/voc.txt >> $datadir/vocMap.txt

#add index number from vocabulary for train, valid, and test
#add </ee> as the end for each sentence in $datadir/data.txt, then combine into one line: $datadir/data_combine.txt
awk '{print $0" (eee) "}' $datadir/train.txt > $datadir/train_AddEnd.txt
awk '{printf $0;}' $datadir/train_AddEnd.txt > $datadir/train_combine.txt

awk '{print $0" (eee) "}' $datadir/valid.txt > $datadir/valid_AddEnd.txt
awk '{printf $0;}' $datadir/valid_AddEnd.txt > $datadir/valid_combine.txt

awk '{print $0" (eee) "}' $datadir/test.txt > $datadir/test_AddEnd.txt
awk '{printf $0;}' $datadir/test_AddEnd.txt > $datadir/test_combine.txt

