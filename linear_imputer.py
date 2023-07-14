# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:45:08 2023
    
    Linear imputer for time-series data
    This is an extension of the work that Skyler Chauff and I have been working
        on last semester (AY23-2).
        
    The next step is to create a Lagrange polynomial with all of the known data
        and use the polynomial to find all of the missing data.
    
@author: grover.laporte
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


class Imputer(object):
    def __init__(self,data,times,admin_cols):
        """
        data - pandas DataFrame
        admin_cols - the number of cols in the data that will not be imputed
        """
        self.times = times
        self.admin_cols = admin_cols
        data.columns = list(data.columns)[:admin_cols]+times
        self.data = data
        #self.X = data.iloc[:,admin_cols:].values
        self.X = pd.DataFrame(data.iloc[:,admin_cols:].values,
                              columns=self.times)
        self.shape = self.X.values.shape
        self.counter = 0
        # rows in the dataframe that have a null value
        rows=self.X.isnull().any(axis=1)
        rows = np.where(rows==True)[0]
        self.rows = rows
        #for each row that has a null value, find the cols that are null
        cols = [] # cols in each row that are null
        for row in rows:
            col_ = np.where(self.X.iloc[row].isnull()==True)[0]
            cols.append(col_)
        self.cols = cols
        self.deleted_rows = []
        
    def consecutive_numbers(self,cols):
        return sorted(cols) == list(range(min(cols),max(cols)+1))
    
    
    def linear(self,df,times):
        """
        method linear >>> used to calculate a missing value given the two
                           nearest points.
        Input: df      >>> the data frame for the two nearest points where 
                            the x-values are in the index and y-values in the
                            values.
               times   >>> list of times with missing values
               
        Output: numpy array for all of the missing times.
        """
        x = df.index
        y = df.values
        pt1 = [x[0],y[0]]; pt2 = [x[1],y[1]]
        m = (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
        b = pt1[1]-m*pt1[0]
        f = lambda t: m*t+b
        ans = []
        for t in times:
            ans.append(int(f(t)))
        return np.array(ans)
    
    def impute_or_not(self,cols):
        """
        use this method to determine whether or not to impute the row
            based on the number of columns that are missing data. More
            is needed to determine how many missing values are too much.
            
            I use a simple formula until that research is done. If there
            are more than three values missing return False, impute otherwise.
        """
        return len(cols)<3
    
    def impute_data(self):
        """
        method is used to impute the data that we want to impute while
            deleting the rows that are missing more than 2 values.
        """
        for i in range(len(self.rows)-1,-1,-1):
            if self.impute_or_not(self.cols[i]):
                row = self.rows[i]
                cols = self.cols[i]
                times,vals = self.values_to_impute(row,cols)
                for t,v in zip(times,vals):
                    self.X.loc[row,t] = v
                if np.any(vals<0):
                    self.deleted_rows.append(row)
                    self.X = self.X.drop(row)
                    self.data = self.data.drop(row)
            else:
                self.deleted_rows.append(self.rows[i])
                self.X = self.X.drop(self.rows[i])
                self.data = self.data.drop(self.rows[i])
                
        data = pd.concat([self.data.iloc[:,:self.admin_cols],
                          self.X],axis=1)
        self.imputed_data = data
        return data
    
    def values_to_impute(self,row,cols):
        """
        this method returns the data points and times of the missing 
            values to pass to the linear imputer.
        Currently, we are passing on data that have more than 2 data points
            missing, so I only check a few possibilities.
            group1 - missing 1 at the start.
            group2 - missing 1 in the middle.
            group3 - missing 1 at the end.
            group4 - missing 2 at the start.
            group5 - missing 2 in the middle.
            group6 - missing 2 at the end.
            group7 - missing 1 at the start and 1 in the middle.
            group8 - missing 1 in the middle and 1 at the end.
            group9 - missing 2 non-consecutive in the middle.
        """
        last_col = self.shape[1]-1
        N = len(cols)
        pts = self.X.iloc[row,:]
        times = np.array(self.times)
        pts_ = 0; t = 0; val = 0; col_times = 0
        if N == 1:
            # only groups 1,2,3
            if 0 in cols:
                #Group1
                t = [times[0]]
                idx = np.array([1,2])
                pts_ = pts.iloc[idx]
                val = self.linear(pts_,t)
                col_times = t
                
            elif last_col in cols:
                #Group3
                t = [times[last_col]]
                idx = np.array([last_col-2,last_col-1])
                pts_ = pts.iloc[idx]
                val = self.linear(pts_,t)
                col_times = t
                
            else:
                #Group2
                col = cols[0]
                t = [times[col]]
                idx = np.array([col-1,col+1])
                pts_ = pts.iloc[idx]
                val = self.linear(pts_,t)
                col_times = t
            
        elif N == 2:
            # can only be in group 4,5,6,7,8
            if self.consecutive_numbers(cols):
                # only groups 4,5,6
                if 0 in cols:
                    #group4
                    t = times[cols]
                    idx = np.array([2,3])
                    pts_ = pts.iloc[idx]
                    val = self.linear(pts_,t)
                    col_times = np.array(t)
                elif last_col in cols:
                    #group6
                    t = times[cols]
                    idx = np.array([last_col-3,last_col-2])
                    pts_ = pts.iloc[idx]
                    val = self.linear(pts_,t)
                    col_times = np.array(t)
                else:
                    #group5
                    t = times[cols]
                    idx = np.array([cols[0]-1,cols[1]+1])
                    pts_ = pts.iloc[idx]
                    val = self.linear(pts_,t)
                    col_times = np.array(t)
                    
            elif 0 in cols:
                #group7
                col = cols[1]
                val = []
                col_times=[]
                ########################################
                t = [times[0]]
                idx = np.array([1,2])
                pts_ = pts.iloc[idx]
                val.append(self.linear(pts_,t)[0])
                col_times.append(t[0])

                ########################################
                t = [times[col]]
                idx = np.array([col-1,col+1])
                pts_ = pts.iloc[idx]
                val.append(self.linear(pts_,t)[0])
                col_times.append(t[0])
                ########################################
                val = np.array(val)
                col_times = np.array(col_times)
                
            elif last_col in cols:
                #group8
                col=cols[0]
                val = []
                col_times=[]
                ########################################
                t = [times[col]]
                idx = np.array([col-1,col+1])
                pts_ = pts.iloc[idx]
                val.append(self.linear(pts_,t)[0])
                col_times.append(t[0])

                ########################################
                t = [times[last_col]]
                idx = np.array([last_col-2,last_col-1])
                pts_ = pts.iloc[idx]
                val.append(self.linear(pts_,t)[0])
                col_times.append(t[0])
                val = np.array(val)
                col_times = np.array(col_times)
            else:
                #group9
                val = []
                col_times=[]
                for col in cols:
                    t = [times[col]]
                    idx = np.array([col-1,col+1])
                    pts_ = pts.iloc[idx]
                    val.append(self.linear(pts_,t)[0])
                    col_times.append(t[0])
                col_times = np.array(col_times)
                
        return col_times,np.array(val)
        
            
st.write("# Linear Imputer for Time-Series data")
st.write("""Ensure that your first column of data in the .csv file is the
         labels for each column and the rows are observations. You need to
         know how many admin columns that you have in your csv file along
         with the times that your data was collected.
         """)
admin_col = st.number_input(":red[Number of non-numerical columns:]",
                      min_value=0,max_value=20,value=2,step=1,
                help="The number of columns used to identify the observation")
time_str = st.text_input("Enter the data collection times: (separated by a comma)",
                         value="0,15,60,90,120,180,240")
time_str=time_str.split(',')
times = [int(time_str[i]) for i in range(len(time_str))]

st.divider()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

try:
    imputer = Imputer(df,times,admin_col)
    new_df = imputer.impute_data()
    row_select=st.selectbox("Select an imputed row to see how the data changed",imputer.rows)
    if row_select not in imputer.deleted_rows:
        data1 = df.loc[row_select,:].iloc[admin_col:]
        data2 = new_df.loc[row_select,:].iloc[admin_col:]
        data = pd.concat([data1,data2],axis=1,ignore_index=True)
        data.columns = ["old","new"]
        data["time"]=data.index
        col1,col2 = st.columns(2)
        c1 = alt.Chart(data,title="Row:"+str(row_select)).mark_point(size=100).encode(
            x="time",
            y="old",
            color=alt.value("salmon"),
            )
        c2 = alt.Chart(data,title="Row:"+str(row_select)).mark_point(size=100).encode(
            x="time",
            y="new",
            color=alt.value("cadetblue"),
            )
        col1.altair_chart(c2+c1,use_container_width=True)
        col2.write("# ")
        col2.write(data)
        st.divider()
        st.download_button(label="Download csv",
                   data = merge_df(df,col),
                   file_name='rename.csv')
    
    
    else:
        st.write("This row was deleted and not used in the final dataframe.")
    
    
    
except:
    st.write("""
             #### Upload a .csv file to be imputed.
             #### Ensure that :red[Non-numerical Columns] are correct
             """)
#print(new_df)
#print("Deleted rows:", imputer.deleted_rows)
