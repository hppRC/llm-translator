wget https://www2.nict.go.jp/astrec-att/member/mutiyama/align/download/para.zip
mv para.zip ./datasets/
unzip ./para.zip -d ./datasets/

wget https://www2.nict.go.jp/astrec-att/member/mutiyama/align/download/align-070215.zip
unzip ./align-070215.zip -d ./datasets/

# wget https://huggingface.co/datasets/sentence-transformers/parallel-sentences/resolve/main/WikiMatrix/WikiMatrix-en-ja-train.tsv.gz
# gunzip WikiMatrix-en-ja-train.tsv.gz
# mv WikiMatrix-en-ja-train.tsv ./datasets/

wget wget https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-ja.tsv.gz
gunzip WikiMatrix.en-ja.tsv.gz
mv WikiMatrix.en-ja.tsv ./datasets/