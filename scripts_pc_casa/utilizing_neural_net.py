def correlation_matrix(df):
    corrMatrix = df.corr()
    sns.heatmap(corrMatrix, annot=True)    
    plt.show()
    return 0

def subtract_avg_scale(df):
    df = df.apply(lambda x:(x-x.mean())/x.var())
    return df

