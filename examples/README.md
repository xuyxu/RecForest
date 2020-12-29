## Reproduce the experiment
1. **Download datasets from ODDS: http://odds.cs.stonybrook.edu/**
    * In the current directory, I have manually added a relatively small dataset **ionosphere** for quick start.
2. **Run the command: `python main.py`**
    * You may need to change the path to the dataset in Line 18 if you want to use other datasets.
    * Recforest will be trained and evaluated over 10 indepdendent trials, and the printing information reports its performance and runtime after each trial.
3. **Compute the average value and standard deviation**
    * Average results from 10 indepdendent trials should match those in the Table 2 and 3 from the original paper (or be very similar because of the different random number generators across platforms).