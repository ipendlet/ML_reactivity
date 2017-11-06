import matplotlib.pyplot as plt
from sklearn.metrics.regression import r2_score


def plotScatterPlot(actual, predicted, outFileName):
    'Make a scatter plot showing the predicted vs actual activation energy for each reaction'
    plt.scatter(actual, predicted, s=7, color='#4b9da6')
    axes = plt.gca()

    # make plot square with equal x and y axes
    bounds = [min(list(actual) + list(predicted) + [0])-1, max(list(actual) + list(predicted))+1]
    plt.axis(bounds * 2)
    axes.set_aspect('equal', adjustable='box')

    # plot the identity for visual reference (10% darker than data)
    plt.plot([bounds[0], bounds[1]], [bounds[0], bounds[1]], color='#d95d41')

    rSquared = r2_score(actual, predicted)
    print(rSquared)
    plt.figtext(0.6,0.15,'$R^2 = $'+format(rSquared,'.4f'), fontsize=11)
    plt.xlabel('QM Calculated $\mathregular{pK_{a}}$', fontsize=10)
    plt.ylabel('Model Predicted $\mathregular{pK_{a}}$', fontsize=10)
    plt.title('Model Predicted vs. QM Calculated $\mathregular{pK_{a}}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(str(outFileName) + '.png')
    plt.clf()
