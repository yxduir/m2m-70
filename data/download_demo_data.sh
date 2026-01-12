data_config=yxdu/srt-demo-s2tt-70
hf download ${data_config} --repo-type dataset --local-dir ./s2tt
if [ -f "./s2tt/audio.tar.gz" ]; then
        echo "解压音频文件..."
        tar -zxvf "./s2tt/audio.tar.gz" -C "./s2tt/"
fi