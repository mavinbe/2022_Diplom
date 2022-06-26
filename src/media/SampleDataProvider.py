from typing import NamedTuple

from media.VideoStreamProvider import VideoStreamProvider


class SampleVideoSpecification(NamedTuple):
    relative_path: str
    fps: float
    minute_to_start: float


sampel_data = {
    "clara_simon_kira_me_hole": [
        SampleVideoSpecification("/data/2022_04_nice/01_clara_simon_kira_me_hole.mp4", 25.04, 0),
        SampleVideoSpecification("/data/2022_04_nice/06_clara_simon_kira_me_hole.mp4", 10.02, 0)
    ],
    "clara_simon_kira_me_last_part": [
        SampleVideoSpecification("/data/2022_04_nice/01_clara_simon_kira_me_last_part.mp4", 25.04, 0),
        SampleVideoSpecification("/data/2022_04_nice/06_clara_simon_kira_me_last_part.mp4", 10.02, 0)
    ]

}


def get_sample_video_specification_by_key(key):
    return sampel_data[key]


def get_video_stream_provider(root_dir, video_specification):
    print(video_specification)
    return VideoStreamProvider(root_dir + video_specification.relative_path, play_back_speed=0.4,
                               fps=video_specification.fps,
                               minute_to_start=video_specification.minute_to_start)
