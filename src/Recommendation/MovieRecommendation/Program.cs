using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

Console.WriteLine("Hello, Movie Recommendation!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane z pliku CSV 
var trainingData = context.Data.LoadFromTextFile<MovieRatingData>("recommendation-ratings-train.csv", hasHeader: true, separatorChar: ',');
var testData = context.Data.LoadFromTextFile<MovieRatingData>("recommendation-ratings-test.csv", hasHeader: true, separatorChar: ',');


var preview = trainingData.Preview();

// 3. Wybór algorytmu 
// dotnet add package Microsoft.ML.Recommender

var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = "userId",
    MatrixRowIndexColumnName = "movieId",
    LabelColumnName = "Label",
   
    NumberOfIterations = 20,
    ApproximationRank = 100
};

var trainer = context.Recommendation().Trainers.MatrixFactorization(options);

// 4. Budowanie modelu
//var pipeline = 
//    context.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
//    .Append(context.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"))
//    .Append(trainer);


// 6. Trenowanie modelu na danych treningowych
var model = trainer.Fit(trainingData);

// 7. Ewaluacja modelu na zbiorze testowym 
//var metrics = context.Regression.Evaluate(testData);

//Console.WriteLine($"R^2: {metrics.RSquared}"); // Współczynnik determinacji R^2
//Console.WriteLine($"MEA: {metrics.MeanAbsoluteError}"); // Średni błąd w jednostkach ceny
//Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}"); // Wskazuje, jak bardzo predykcje odbiegają od rzeczywistych wartości


// 8. Predykcja dla nowego przypadku
var newMovieRatingData = new MovieRatingData { userId = 1, movieId = 10 };

// 9. Tworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<MovieRatingData, MovieRatingPrediction>(model);


// 10. Predykcja 
var prediction = predictionEngine.Predict(newMovieRatingData);


if (prediction.Score > 3.5f)
    Console.WriteLine($"Movie {newMovieRatingData.movieId} is recommended for {newMovieRatingData.userId} with Rating {prediction.Score:F2}");
else
    Console.WriteLine($"Movie {newMovieRatingData.movieId} is not recommended for {newMovieRatingData.userId} with Rating {prediction.Score:F2}");

Console.ReadLine();

public class MovieRatingData
{
    [LoadColumn(0)]
    [KeyType(count: 262111)]
    public uint userId { get; set; }

    [LoadColumn(1)]
    [KeyType(count: 262111)]
    public uint movieId { get; set; }

    [LoadColumn(2), ColumnName("Label")]
    public float rating { get; set; }
}

public class MovieRatingPrediction
{
    public float Score { get; set; }

}