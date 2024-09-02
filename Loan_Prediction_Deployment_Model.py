#Loan Prediction Deployment Model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import statistics
import streamlit as st

#Let us set our background color
st.markdown(
    """
    <style>
        .main {
            background-color: #0c224e;
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #a7aac5;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<b><u><center>LOAN STATUS PREDICTION</center></u></b>",unsafe_allow_html=True)
st.markdown("This project is intended for learning Purposes")
@st.cache_resource
def data():
    url = "https://github.com/jackonyango2545/Loan-Status-Prediction/raw/main/loan_approval_dataset.csv"
    df = pd.read_csv(url)
   
    def outliers(x):
        Arithmetic_mean = statistics.mean(x)
        Standard_Deviation = statistics.stdev(x)
        upper = Arithmetic_mean + (3*Standard_Deviation)
        lower = Arithmetic_mean - (3*Standard_Deviation)

        outlier = df[(x<lower) | (x>upper)]

        if len(outlier) > 0:
            df.drop(df[x < lower].index, inplace=True)
            df.drop(df[x > upper].index, inplace=True)

            Arithmetic_mean = statistics.mean(x)
            Standard_Deviation = statistics.stdev(x)
            upper = Arithmetic_mean + (3*Standard_Deviation)
            lower = Arithmetic_mean - (3*Standard_Deviation)
            outlier = df[(x<lower) | (x>upper)]
        del outlier
        del lower
        del upper
        del Arithmetic_mean
        del Standard_Deviation


    outliers(df['income_annum'])
    outliers(df['loan_amount'])
    outliers(df['residential_assets_value'])
    outliers(df['commercial_assets_value'])
    outliers(df['luxury_assets_value'])
    outliers(df['bank_asset_value'])
    outliers(df['cibil_score'])
    #Using online search, a credit score cannot be greater than 850
    df.drop(df[df['cibil_score']>850].index, inplace=True)
    #Lets rename some columns so that someone can easily understand them and to add a new column
    try:
        df.rename(columns={'education':'Education Level','income_annum':'Annual Income','cibil_score':'Credit Score','loan_status':'Loan Status'},inplace=True)
    except:
        print("The column names were already updated")
    finally:
        df['Annual Income to Loan Amount Ratio'] = round(((df['Annual Income'])/(df['loan_amount'])),4)
        df['Residential Value to Loan Ratio'] = (df['residential_assets_value']/df['loan_amount'])
        df['Commercial Value to Loan Ratio'] = (df['commercial_assets_value']/df['loan_amount'])
        df['Luxury Value to Loan Ratio'] = (df['luxury_assets_value']/df['loan_amount'])
        df['Bank Value to Loan Ratio'] = (df['bank_asset_value']/df['loan_amount'])
    return df

df = data()

st.write("""**ABOUT THE PROJECT**\n
When applying for a loan, sometimes we get the anxiety of not knowing whether the loan is going to be approved or not.This project is 
intended to help someone understand what factors tends to affect their loan status and probability of their loan being approved
""")

st.write("""**Objective**\n
The aim of this project is to come up with a model that can be used to predict and make the decision of whether a 
loan application should be approved or rejected based on the various features selected\n
I used this dataset because of the various features that it has that I believe has a relationship with the loan 
status. The Dataset is obtained from Kaggle link https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset ,
this includes columns like""",list(df.columns))

st.write("The brief view of the dataset is:")
st.write(df.head(5))

st.write("""**RESEARCH QUESTIONS**\n
- Does credit score have a relationship with the loan approval?
- Is there a relationship between income level, loan status, and credit score?
- Do factors like the number of dependents, education level, and self-employment relate to credit 
score and loan approval?
- What is the average loan-to-income ratio? Does the loan-to-income ratio influence the loan approval status?
- Does the loan term and amount (conditional probability) affect the approval status?
- Is there a relationship between the asset value and a person’s self-employment status, income, and loan amount? 
- Does an individual's asset value affect their loan status
""")


st.write("""**RESULT**\n
During the analysis, I utilized various statistical modeling analysis techniques such as ANOVA and T_Test. 
This helped me gain insights and come with the following as my result""")

st.write("One of the factors affecting the loan status is:")

st.write("""***1. Loan Term***\n
From the data we have used for the analysis, we can see that the loan term has a slight influence over loan approval. 
It shows that people who want to pay back their loans in less than 8 years tend to have a better chance of being 
accepted for the loan. This is because a shorter waiting period reassures the lender that the loan will be repaid 
quickly with some interest.However, choosing a shorter loan term does not guarantee approval, and a longer loan 
term does not guarantee rejection. This is illustrated in the graph below.
""")


@st.cache_resource
def foto():
    df1=pd.DataFrame(df.groupby(['loan_term','Loan Status'])['loan_id'].count())
    df1.reset_index(inplace=True)

    fig, ax = plt.subplots()
    sns.lineplot(x='loan_term',y='loan_id',data=df1,hue='Loan Status',palette='deep',ax=ax,legend=True)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2))
    plt.grid()
    plt.title("Distribution of Loan status over the loan term")
    plt.xlabel('Loan Term in Years')
    plt.ylabel('Number of Loan Status')
    return st.pyplot(fig)

foto()


st.write("""***2. Credit Score***\n
According to how credit scores are calculated, people with higher credit scores show a higher likelihood of 
repaying their loans, thereby being considered creditworthy. A credit score above 600 is generally considered good, 
with scores ranging from 600 to 850, where 850 is the maximum credit score. People with lower credit scores, below 
500, tend to have a higher chance of their loans not being approved. This is illustrated in the box plot below.
""")


@st.cache_resource
def foto2():
    fig, ax = plt.subplots()
    plt.grid()
    sns.boxplot(x='Loan Status', y='Credit Score', data=df, palette='deep', hue='Loan Status', ax=ax,legend=False)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=50))
    plt.title("Distribution of Credit score on Loan Status")
    return st.pyplot(fig)

foto2()

st.write("""***3. Annual Income to Loan Amount Ratio***\n
This is the ratio of the annual income over the loan amount one is requesting for. As much as loan amount and 
someones income doesn't display a relationship with the loan status, this ratio tends to show that someones loan 
amount to income ratio have an implication over the amount of money one is intending to borrow. This can be because 
this ratio tends to show the ability of someone paying back the loan.""")

@st.cache_resource
def foto3():
    fig, ax = plt.subplots()
    sns.boxplot(x='Loan Status', y='Annual Income to Loan Amount Ratio', data=df, palette='deep', hue='Loan Status', ax=ax, legend=False)
    plt.grid()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(base=0.05))
    plt.title("Distribution of Annual Income to Loan Amount Ratio on Loan Status")
    return st.pyplot(fig)

foto3()

st.write("""
**Model**\n
I decided to use ***Logistic Regression Model*** because of the nature of the problem we are intending to solve. 
Logistic Regression Model is used especially in "decision making scenarios" this is because it uses the concept of probability 
to decide on what decision it will decide on.
""")

@st.cache_resource
def changer():
    def changing_values(x):
        if x == ' Approved':
            return 1
        elif x ==' Rejected':
            return 0
        else:
            return x
    df['Loan Status'] = df['Loan Status'].apply(changing_values)
    return exit

changer()  

@st.cache_resource
def MyModel():
    y = df['Loan Status']
    x = df[['Annual Income to Loan Amount Ratio','Credit Score','loan_amount','loan_term']]

    Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size = 0.2,random_state=42)
    return Xtrain,Xtest,Ytrain,Ytest

Xtrain,Xtest,Ytrain,Ytest = MyModel()

x,y = st.columns([1,1])
with x:
    st.write("**Hyper Parameter Tuning.\n*Adjust The Hyperparameters to your interests***")
    options = [0.001,0.01,0.1,1,10,50,100]
    Value = st.select_slider("Choose the value of hyperparameter C\n",options=options,value=1)
    Max_iter_value = st.slider("Choose the Maximum Number of iterations you want\n",min_value=50,max_value=500,value=100,step=50,format='%d')  
    options2 =[0.00001,0.0001,0.001,0.005,0.01,0.1]
    tolerance_value = st.select_slider("Choose the tolerance Level\n",options=options2,value=0.0001)

    model = LogisticRegression(solver='liblinear',C=Value,max_iter=Max_iter_value,penalty='l1',tol=tolerance_value,class_weight='balanced',random_state=42)
    model.fit(Xtrain,Ytrain)
    Ypredict = model.predict(Xtest)
    a = st.button("Test")

with y:
    st.write("**Your Hyperparameters have the following effects on performance score**")
    if a:
        st.write("The Accuracy score is\n",accuracy_score(Ypredict,Ytest))
        st.write("The Effect of change on the hyperparameters score on Approved Loan is")
        st.write("The Precision score is\n",precision_score(Ypredict,Ytest,pos_label= 1))
        st.write("The F1_Score is\n",f1_score(Ypredict,Ytest,pos_label= 1))
        st.write("The Effect of change on the hyperparameters score on Rejected Loan is")
        st.write("The Precision score is\n",precision_score(Ypredict,Ytest,pos_label= 0))
        st.write("The F1_Score is\n",f1_score(Ypredict,Ytest,pos_label= 0))

st.write("**MODEL DEPLOYMENT**")
st.write("For someone to be able to be eligible to borrow a loan, they need to provide various information")

a,b = st.columns([1.5,1])

with a:
    Annual_Income = st.number_input("Annual Income in USD",min_value=1)
    Loan_Amount = st.number_input("Loan Amount in USD",min_value = 1)
    Credit_Score = st.number_input("Credit Score",min_value=300,max_value=850,value=575)
    Loan_Term = st.number_input("Loan Term in Years",max_value=30,min_value=1)
    t = st.button("Predict")

with b:
    if t:
        def outPut():
            ratio1 = Annual_Income/Loan_Amount
            value_to_predict =  pd.DataFrame({
                'Annual Income to Loan Amount Ratio': [ratio1],
                'Credit Score': [Credit_Score],
                'loan_amount': [Loan_Amount],
                'loan_term': [Loan_Term]
            })
            prediction = model.predict(value_to_predict)
            prediction_proba = model.predict_proba(value_to_predict)

            if prediction[0] == 1:
                prediction_text = '<span style="color:green">Approved</span>'
            else:
                prediction_text = '<span style="color:red">Rejected</span>'

            prediction_Made = st.markdown(f'Your Loan is Likely to be {prediction_text} check the probability matrix for the likelyhoods', unsafe_allow_html=True)
            probability = st.write('\nPrediction Probability Outcome', prediction_proba)
            return prediction_Made, probability
        
        outPut()

st.write("""
**LIMITATIONS**\n
- Not everyone has a credit card or know how access their credit score. Also for new clients, it might be hard to determine their credit value\n
- Its hard for someone to borrow a loan to start a business if they are starting from unemployment or have no source of income
- The datset has less number of records and can be biased on a certain group of people.
""")
