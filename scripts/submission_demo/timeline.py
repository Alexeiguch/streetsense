"""Editable film script. Times are seconds on the three-minute master."""
DURATION=180
SHOTS=[
 (0,7,'hook','vehicle','GREEN.','STILL WAIT.'),
 (7,17,'problem','tyumen','THE STREET.','EVERY DETAIL MATTERS.'),
 (17,28,'intro','sidewalk','StreetSense.',''),
 (28,40,'track','tyumen','TRACK','WHAT MOVES.'),
 (40,51,'red','dc_red','RED HAND.','DO NOT CROSS.'),
 (51,67,'walk','dc_walk','TWO CHECKS.','ONE CROSSING CUE.'),
 (67,82,'disagree','tyumen','AI SAYS YES.','STILL NO GO.'),
 (82,96,'countdown','countdown','ONE SECOND LEFT.','DO NOT START.'),
 (96,109,'europe','europe','DIFFERENT SIGNALS.','LOCAL CONTEXT.'),
 (109,122,'sam','tyumen','FIND THE','CROSSWALK.'),
 (122,136,'architecture','tyumen','SEE. REASON.','SPEAK.'),
 (136,149,'freshness','dc_walk','OLD EVIDENCE?','LET IT EXPIRE.'),
 (149,159,'3d',None,'TEST THE FRONTIER.','KEEP WHAT HELPS.'),
 (159,173,'payoff','tyumen','MORE CONTEXT.','LESS GUESSWORK.'),
 (173,180,'end',None,'StreetSense.','HEAR WHAT MATTERS.'),
]
COPY=[
 (.45,5.7,"Green. So why won't it say go?",'narrator'),
 (7.4,16.2,'A crossing. A moving car. A tiny signal. For a blind or low vision pedestrian, one missing detail matters.','narrator'),
 (17.35,27.1,'Street Sense turns a camera feed into short spoken street cues. Built for blind and low vision pedestrians.','narrator'),
 (28.35,39.2,'Local computer vision tracks people and vehicles, finds crossings, and reads pedestrian signals. The fast loop keeps watching while Meta A.I. reasons in the background.','narrator'),
 (40.35,48.2,'Start with a red hand. The local reader sees red. Muse Spark agrees. The answer is simple.','narrator'),
 (48.5,50.5,'Do not cross.','cue'),
 (51.35,62.0,'Now the walk signal. Local vision reads walk. Muse Spark reads the same image and returns yes. Only both together unlock the crossing cue.','narrator'),
 (62.3,66.7,'Walk signal looks clear. Still listen and check.','cue'),
 (67.35,79.3,"Here's the test that matters. Muse Spark says yes. The local reader is unclear. Street Sense refuses a go cue. Model confidence alone is not permission.",'narrator'),
 (79.7,81.7,'Do not cross.','cue'),
 (82.35,93.0,"And when the countdown is almost gone, a green looking light isn't enough either. One second left: do not start crossing.",'narrator'),
 (93.4,95.6,'Do not cross.','cue'),
 (96.35,108.1,'Signals change by country. The UK and Europe profile looks for green walking figures, with a closer pass for oversized signal heads.','narrator'),
 (109.35,121.1,"Meta's Segment Anything gives pixel level masks for crosswalks and signal heads. Geometry stays advisory. A mask never becomes crossing permission.",'narrator'),
 (122.35,135.1,'The architecture stays simple. Computer vision for motion and signals. Muse Spark for scene reasoning. Segment Anything for grounded masks. Short, prioritized speech for the person.','narrator'),
 (136.35,148.1,'We designed for uncertainty, not just happy paths. Old model replies expire. Camera resets clear old state. Unclear evidence always means wait.','narrator'),
 (149.35,158.35,'We tested Meta Map Anything too. Real 3D output, but seconds per batch. It stays offline, outside the live decision loop.','narrator'),
 (159.35,172.1,"Street Sense is a prototype, built around one rule: never turn uncertainty into permission. Next, accessibility led evaluation with the people it's designed for.",'narrator'),
 (173.35,179.3,'Street Sense. Hear what matters. Explore the prototype.','narrator'),
]
# Complete spoken beats, with shorter holds for social viewing; 50 seconds.
SOCIAL=[(0,4),(17,26),(51,67),(67,82),(173,179)]
