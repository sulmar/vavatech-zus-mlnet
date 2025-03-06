
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane z pliku CSV za pomocą Loadera, dzięki temu nie musimy stosować atrybutów

//var loader = context.Data.CreateTextLoader(new[]
//{
//    new TextLoader.Column("ID", DataKind.String, 0),
//    new TextLoader.Column("Area", DataKind.String, 1),
//    new TextLoader.Column("Title", DataKind.String, 2),
//    new TextLoader.Column("Description", DataKind.String, 3),
//}, hasHeader: true, separatorChar: '\t');

// var trainingData = loader.Load("core-issues.csv");

var trainingData = context.Data.LoadFromTextFile<GitHubIssueData>("core-issues.csv", hasHeader: true, separatorChar: '\t');

var preview = trainingData.Preview();

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainingData, testFraction: 0.2f);


// 4. Wybór algorytmu 
var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy();

// 5. Budowanie modelu
var mapValueToKey = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "Label");
var featuredTitle = context.Transforms.Text.FeaturizeText(outputColumnName: "FeaturedTitle", inputColumnName: "Title");
var featuredDescription = context.Transforms.Text.FeaturizeText(outputColumnName: "FeaturedDescription", inputColumnName: "Description");
var features = context.Transforms.Concatenate( "Features", "FeaturedTitle", "FeaturedDescription");
var mapKeyValue = context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel");

var pipeline = mapValueToKey
    .Append(featuredTitle)
    .Append(featuredDescription)
    .Append(features)    
    .Append(trainer)
    .Append(mapKeyValue)
    ;

// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(testTrainSplit.TrainSet);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testTrainSplit.TestSet);

var metrics = context.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}"); 
Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}"); 
Console.WriteLine($"LogLoss: {metrics.LogLoss}");
Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");


// 8. Tworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<GitHubIssueData, GitHubIssueDataPrediction>(model);


// 9. Predykcja dla nowego przypadku

while (true)
{
    Console.Write("Title > ");
    var title = Console.ReadLine();

    Console.Write("Description > ");
    var description = Console.ReadLine();

    var newIssueData = new GitHubIssueData
    {
        Title = title,
        Description = description
    };

    // 10. Predykcja obszaru na podstawie opisu zgłoszenia
    var prediction = predictionEngine.Predict(newIssueData);

    // 11. Wyświetlenie wyniku predykcji
    Console.WriteLine($"Title: {newIssueData.Title}");
    Console.WriteLine($"Description: {newIssueData.Description}");
    Console.WriteLine($"Predicted Area: {prediction.PredictedArea ?? "NULL"}");

}


Console.ReadLine();

public class GitHubIssueData
{
    [LoadColumn(0)]
    public string ID { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public string Area { get; set; }

    [LoadColumn(2)] 
    public string Title { get; set; }

    [LoadColumn(3)]
    public string Description { get; set; }
}

public class GitHubIssueDataPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedArea { get; set; }
}