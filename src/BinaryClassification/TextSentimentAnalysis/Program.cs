
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytanie danych z pliku csv
var trainData = context.Data.LoadFromTextFile<CommentData>("gov-comments.csv", separatorChar: ',', hasHeader: true, allowQuoting: true);

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainData, testFraction: 0.2f);
var trainingData = testTrainSplit.TrainSet;
var testData = testTrainSplit.TestSet;

// 4.  Wybór algorytmu - wybieram Catalog klasyfikacji binarnej 
var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

// 5. Budowanie modelu
var pipeline = 
    context.Transforms.Text.FeaturizeText("FeaturizedComment", "Comment")
    .Append(context.Transforms.Concatenate("Features", "FeaturizedComment"))
    .Append(trainer);

// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(trainingData);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testData);

var metrics = context.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score}");

var predictionRows = context.Data.CreateEnumerable<CommentPrediction>(predictions, reuseRowObject: false);
var trueCount = predictionRows.Count(p => p.PredictedComment);
var falseCount = predictionRows.Count(p => !p.PredictedComment);
Console.WriteLine($"Predictions True: {trueCount} False: {falseCount}");

// 9. Tworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<CommentData, CommentPrediction>(model);

while (true)
{
    Console.Write("> ");
    var comment = Console.ReadLine();

    var newComment = new CommentData { Comment = comment };

    var prediction = predictionEngine.Predict(newComment);

    var result = prediction.PredictedComment ? "Pozytywny" : "Negatywny";

    // 11. Wyświetlenie wyniku predykcji
    Console.WriteLine($"Prediction: {result}");
    Console.WriteLine($"Probalitity: {prediction.Probability:P2}");
    Console.WriteLine($"Score: {prediction.Score}");


}



public class CommentData
{
    [LoadColumn(0)]
    public string Comment { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}

public class CommentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedComment { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
