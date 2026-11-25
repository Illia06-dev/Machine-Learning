## Title
Real-time gym exercise recognition and repetition counting on a smartphone
## Dataset
**MM-Fit** (Strömbäck, Huang, Radu, ACM IMWUT 2020), https://mmfit.github.io: inertial data (accelerometer, gyroscope)
from smartphones, smartwatches and an earbud, recorded during full-body workouts; 10 participants, 10 exercises
(squats, push-ups, dumbbell shoulder press, lunges, dumbbell rows, sit-ups, triceps extensions, bicep curls,
lateral raises, jumping jacks) plus a "no exercise" class, over 800 minutes in total.
- Split: by participant, so the test participants are never seen in training.
- Personal test set: my own workouts recorded with my phone in the pocket (phyphox / Sensor Logger, CSV).
- License: research dataset, used for education with citation of the paper.
- Known limits: only 10 people, recorded in a lab; my phone, its position and my technique differ from the dataset.