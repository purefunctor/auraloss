from dataclasses import dataclass
from typing import List
from enum import Enum


# class syntax
class Position(Enum):
    CLOSE = 1
    MID = 2
    FAR = 3

class MX20(Enum):
    TWO = 2
    FOUR = 4
    EIGHT = 8
    TWELVE = 12

class Eleven78(Enum):
    FOUR = 4
    EIGHT = 8
    TWELVE = 12

class Row(Enum):
    CLOSE = 1
    MID = 2
    FAR = 3


def position_and_row_to_cm(position: Position, row: Row) -> int:
    return (
        20 if position == Position.CLOSE else 25 if position == Position.MID else 30
    ) + (17.5 * (0 if row == Row.CLOSE else 1 if row == Row.MID else 2))


@dataclass
class DistanceMarker:
    nsamples: int
    singer: str
    position: Position
    row: Row

    def __repr__(self) -> str:
        return f"At sample {self.nsamples}, singer name: {self.singer}, position {self.position.name}, mic row {self.row.name}, distance mic<->singer: {position_and_row_to_cm(position=self.position, row=self.row)}"


@dataclass
class Experiment1AudioFile:
    file_name: str
    mic: str
    markers: List[DistanceMarker]
    sample_rate: int = 44100
    bit_depth: int = 24

    def __repr__(self) -> str:
        return "\n@@@@\n".join([repr(x) for x in self.markers])


RAW_EXPORT_DAY_1 = """
1   	0:08         	352800            	Samples  	HELLEKA                          	
16  	7:02         	18653024          	Samples  	10:44                            	
17  	19:51        	52546144          	Samples  	10:58                            	
18  	41:04        	108662400         	Samples  	11:40                            	
2   	52:30        	138936320         	Samples  	HANNA 12:00                      	
3   	71:13        	188460620         	Samples  	12:46                            	
4   	97:10        	257124320         	Samples  	13:15                            	
5   	131:24       	347722072         	Samples  	AKU 14:00                        	
6   	148:45       	393601841         	Samples  	TAJUAMINENÊAIKAKOODEISTA         	
7   	159:53       	423088472         	Samples  	14:50                            	
8   	176:31       	467095896         	Samples  	15:10                            	
9   	194:07       	513659224         	Samples  	15:40                            	
10  	204:55       	542232920         	Samples  	15:55                            	
11  	218:59       	579457368         	Samples  	MICHAEL 16:00                    	
12  	232:29       	615174488         	Samples  	16:35                            	
13  	238:26       	630903128         	Samples  	16:45                            	
14  	245:05       	648510377         	Samples  	16:50                            	
15  	257:34       	681562456         	Samples  	MIKE 17:15 25cm                  	
19  	265:47       	703304024         	Samples  	17:30 
"""

RAW_EXPORT_DAY_2 = """
M A R K E R S  L I S T I N G
#   	LOCATION     	TIME REFERENCE    	UNITS    	NAME                             	COMMENTS
1   	0:03         	133120            	Samples  	SIINA 10:00                      	
2   	3:28         	9175040           	Samples  	1178 A not working               	
3   	18:47        	49741824          	Samples  	10:45                            	
4   	35:45        	94633984          	Samples  	11:05                            	
5   	57:22        	151814144         	Samples  	11:30                            	
6   	80:31        	213090304         	Samples  	11:55                            	
7   	99:16        	262668288         	Samples  	12:20                            	
8   	120:28       	318767104         	Samples  	12:50                            	
9   	141:55       	375521280         	Samples  	AMANDA 13:30                     	
10  	153:36       	406454272         	Samples  	13:50                            	
11  	167:25       	443023360         	Samples  	14:10                            	25 cm
12  	172:48       	457268493         	Samples  	14:18                            	CM1A 1 - Fast A, Med Fast R, 3:1
    	             	                  	           	                                 	CM1A 2 - SlowA, Med R, 4:1
    	             	                  	           	                                 	
    	             	                  	           	                                 	MX-20 8:1 and 12:1
    	             	                  	           	                                 	
    	             	                  	           	                                 	1178: 12:1, 1 A, 3 R
13  	192:47       	510132224         	Samples  	14:45                            	
14  	215:11       	569376768         	Samples  	ERIKA 15:15                      	
15  	229:32       	607387648         	Samples  	15:35                            	
16  	249:54       	661258240         	Samples  	16:00                            	
17  	260:12       	688521216         	Samples  	VILJA 16:20                      	
18  	275:34       	729153536         	Samples  	16:50                            	
19  	278:19       	736428032         	Samples  	16:55                            	
20  	305:23       	808058880         	Samples  	SEVERI 17:30                     	
21  	317:46       	840826880         	Samples  	17:45                            	
22  	331:59       	878444544         	Samples  	18:00                            	
23  	346:28       	916783104         	Samples  	18:20                            	
24  	360:52       	954859520         	Samples  	18:40                            	
25  	367:45       	973078528         	Samples  	18:45 30 cm                      	
26  	378:46       	1002242048        	Samples  	VAIHTO          
"""


def annotate(l):
    O = []
    N = None
    for x in l:
        if isinstance(x, str):
            N = x
        else:
            O.append((N, x))
    return O


def parse_raw_samples(S):
    return [
        int([y for y in x.split(" ") if y != ""][2])
        for x in S.split("\n")
        if "Samples" in x
    ]


RAW_SAMPLES_DAY_1 = parse_raw_samples(RAW_EXPORT_DAY_1)
RAW_DISTANCES_DAY_1 = annotate(
    [
        "Hellekka",
        Position.FAR,
        Position.FAR,
        Position.FAR,
        Position.CLOSE,
        "Hanna",
        Position.CLOSE,
        Position.CLOSE,
        Position.MID,
        "Aku",
        Position.FAR,
        Position.FAR,
        Position.FAR,
        Position.MID,
        Position.MID,
        Position.CLOSE,
        "Michael",
        Position.FAR,
        Position.MID,
        Position.MID,
        Position.CLOSE,
        "Mike",
        Position.MID,
        Position.CLOSE,
    ]
)

RAW_COMPRESSOR_DAY_1 = annotate(
    [
        "Hellekka",
        (MX20.TWO, None),
        (MX20.FOUR, None),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        "Hanna",
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.TWO, MX20.FOUR),
        "Aku",
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        "Michael",
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        "Mike",
        (None, None), #??????
        (None, None), #??????
    ]
)
RAW_1178_DAY_1 = annotate(
    [
        "Hellekka",
        (Eleven78.FOUR, 3, 7),
        (Eleven78.EIGHT, 3, 7),
        (Eleven78.TWELVE, 3, 7),
        (Eleven78.TWELVE, 3, 7),
        "Hanna",
        (Eleven78.FOUR, 7, 5),
        (Eleven78.EIGHT, 1, 1),
        (Eleven78.TWELVE, 2, 5), ### PERHAPS NONE??
        "Aku",
        (Eleven78.FOUR, 2, 7),
        (Eleven78.FOUR, 2, 7),
        (Eleven78.EIGHT, 1, 7),
        (Eleven78.EIGHT, 1, 7),
        (Eleven78.FOUR, 7, 5),
        (Eleven78.FOUR, 7, 5),
        "Michael",
        (Eleven78.FOUR, 7, 5),
        (Eleven78.FOUR, 7, 5),
        (Eleven78.EIGHT, 4, 7),
        (Eleven78.EIGHT, 4, 7),
        "Mike",
        None, #??????
        None, #??????
    ]
)
assert len(RAW_SAMPLES_DAY_1) == 19
assert len(RAW_SAMPLES_DAY_1) == len(RAW_DISTANCES_DAY_1)
assert len(RAW_COMPRESSOR_DAY_1) == len(RAW_DISTANCES_DAY_1)
assert len(RAW_1178_DAY_1) == len(RAW_DISTANCES_DAY_1)

RAW_SAMPLES_DAY_2 = parse_raw_samples(RAW_EXPORT_DAY_2)
RAW_DISTANCES_DAY_2 = annotate(
    [
        "Siina",
        Position.FAR,
        Position.FAR,
        Position.FAR,
        Position.MID,
        Position.MID,
        Position.CLOSE,
        Position.CLOSE,
        Position.CLOSE,
        "Amanda",
        Position.CLOSE,
        Position.CLOSE,
        Position.MID,
        Position.MID,
        Position.FAR,
        "Eerika",
        Position.FAR,
        Position.MID,
        Position.CLOSE,
        "Vilja",
        Position.CLOSE,
        Position.FAR,
        Position.FAR,
        "Severi",
        Position.CLOSE,
        Position.CLOSE,
        Position.CLOSE,
        Position.MID,
        Position.MID,
        Position.FAR,
        "Mirjam",
        Position.MID,
    ]
)
RAW_COMPRESSOR_DAY_2 = annotate(
    [
        "Siina",
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (None, None), #??????
        "Amanda",
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.TWO, MX20.FOUR),
        "Eerika",
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        "Vilja",
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.TWO, MX20.FOUR),
        "Severi",
        (None, None),
        (None, None),
        (MX20.TWO, MX20.FOUR),
        (MX20.TWO, MX20.FOUR),
        (MX20.EIGHT, MX20.TWELVE),
        (MX20.EIGHT, MX20.TWELVE),
        "Mirjam",
        (None, None),
    ]
)
RAW_1178_DAY_2 = annotate(
    [
        "Siina",
        (Eleven78.EIGHT, 4, 7),
        (Eleven78.EIGHT, 4, 7),
        (Eleven78.EIGHT, 1, 1),
        (Eleven78.EIGHT, 1, 1),
        (Eleven78.FOUR, 2, 4),
        (Eleven78.FOUR, 2, 4),
        (Eleven78.EIGHT, 3, 5),
        (Eleven78.FOUR, 3, 7),
        "Amanda",
        (Eleven78.FOUR, 3, 7),
        (Eleven78.TWELVE, 1, 3),
        (Eleven78.TWELVE, 1, 3),
        (Eleven78.TWELVE, 1, 3),
        (Eleven78.EIGHT, 4, 4),
        "Eerika",
        (Eleven78.EIGHT, 4, 4),
        (Eleven78.TWELVE, 2, 2),
        (Eleven78.TWELVE, 2, 2),
        "Vilja",
        (Eleven78.TWELVE, 2, 2),
        (Eleven78.TWELVE, 2, 2),
        (Eleven78.FOUR, 4, 6),
        "Severi",
        None,
        None,
        (Eleven78.TWELVE, 7,7),
        (Eleven78.TWELVE, 7,7),
        (Eleven78.FOUR, 6,6),
        (Eleven78.FOUR, 6,6),
        "Mirjam",
        None,
    ]
)
assert len(RAW_SAMPLES_DAY_2) == len(RAW_SAMPLES_DAY_2)
assert len(RAW_SAMPLES_DAY_2) == len(RAW_DISTANCES_DAY_2)

DAY_1_AUDIO_FILES = [
    ##
    Experiment1AudioFile(
        file_name="67_near.wav",
        mic="67",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="269_far.wav",
        mic="269",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="414_near.wav",
        mic="414",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="414_far.wav",
        mic="414",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="87_near.wav",
        mic="87",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="87_far.wav",
        mic="87",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_near.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_middle.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.MID
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_far.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_1, *zip(*RAW_DISTANCES_DAY_1)
            )
        ],
    ),
]

assert len(DAY_1_AUDIO_FILES) == 9


DAY_2_AUDIO_FILES = [
    ##
    Experiment1AudioFile(
        file_name="67_near.wav",
        mic="67",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="269_far.wav",
        mic="269",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="414_near.wav",
        mic="414",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="414_far.wav",
        mic="414",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="87_near.wav",
        mic="87",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="87_far.wav",
        mic="87",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_near.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.CLOSE
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_middle.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.MID
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
    ##
    Experiment1AudioFile(
        file_name="103_far.wav",
        mic="103",
        markers=[
            DistanceMarker(
                nsamples=nsamples, singer=singer, position=position, row=Row.FAR
            )
            for nsamples, singer, position in zip(
                RAW_SAMPLES_DAY_2, *zip(*RAW_DISTANCES_DAY_2)
            )
        ],
    ),
]

assert len(DAY_2_AUDIO_FILES) == 9

if __name__ == "__main__":
    # print("DAY 1")
    # print("******************")
    # for x in DAY_1_AUDIO_FILES:
    #     print(x)
    #     print("##################")
    # print("DAY 2")
    # print("******************")
    # for x in DAY_2_AUDIO_FILES:
    #     print(x)
    #     print("##################")
    # print(RAW_SAMPLES_DAY_1)
    pass