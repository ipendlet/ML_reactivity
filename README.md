#Overview of the code files contained in the prediction of chemical reactivity with machine learning project
main.py: functions for data parsing, executing machine learning algorithms, and plotting results

Reaction.py: Reaction class which is a representation of the data associated with an individual reaction and has the ability to construct various feature representations of itself

DrivingCoordinate.py: DrivingCoordinate class which represents the data associated with an individual add or break move

#Next steps from Mina and Josh talking to Paul on 04/29/16
Try a different chemical data set than the one we had already tried

  Try training on one dataset and predicting on another dataset to see how broadly applicable the learned correlations are
  
Examine the reactions that were very poorly predicted and see if there is something that they have in common to gain understanding about when the predictor fails

In the context of SVR, how to extract which features turned out to be more valuable as predictors and which ones were less valuable

Could try a neural network starting with a single hidden layer

When using kernel methods or others where the distance or similarity between data points is important, think about what would be appropriate metrics for computing similarity or distance between reactions with different numbers of add / break moves
