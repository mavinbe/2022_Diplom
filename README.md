This file gives an overal overview.

In Log.md u can find a hopefully consistent and daily work log.


## run

to activate env
```bash
conda activate diplom
```

to run pipeline
```bash
python src/app/run_pipeline.py
```


to run tests
```bash
pytest
```



### tooling

extract split from video
```ffmpeg
ffmpeg -ss 00:29:39.0 -i XXXX1.mp4 -c copy -t 00:10:00.0 XXXX_split_1.mp4

```



TODO 
- catch "global" errors in "run()" to restart process, so we can restore from crashing errors
- check system heat
- pause just after reset
- smoothern movement
- test mit 4 videos


REFACTOR BEFORE continue

