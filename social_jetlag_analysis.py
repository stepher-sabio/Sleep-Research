"""
social_jetlag_analysis.py
Comprehensive weekday/weekend and social jet lag analysis

Research Questions:
1. Do sleep patterns differ between weekdays vs weekends?
2. What is the magnitude of social jet lag in toddlers?
3. Does Monday show recovery patterns after weekends?
4. Does social jet lag vary by age group?

Social Jet Lag (SJL) = |Weekend sleep midpoint - Weekday sleep midpoint|
Sleep Midpoint = (Bedtime + Wake time) / 2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# Configuration
input_folder = '/Users/stepher/Desktop/Actigraphy2/data