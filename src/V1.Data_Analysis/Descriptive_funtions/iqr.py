from .percentile import percentile

""" 
The Interquartile Range (IQR) is a statistical measure 
of variability that describes the spread of the middle 50% 
of a dataset. It is calculated by subtracting the first 
quartile (Q1, the 25th percentile) from the 
third quartile (Q3, the 75th percentile).


A small IQR means the grades for the bulk of the students are very 
consistent and clustered together.

A large IQR means there is more variation in performance, 
even among the average students.
"""
def iqr(values):
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    return iqr

if __name__ == "__main__":
    iqr()