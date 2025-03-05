
using Microsoft.ML;
using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations.Schema;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, Car Prices Prediction!");

        // dotnet add package Microsoft.ML

        // 1. Tworzymy kontekst ML.NET
        var context = new MLContext();

        // 2. Wczytujemy dane z pliku CSV
        var trainingData = context.Data.LoadFromTextFile<CarData>("car-model-prices.csv", hasHeader: true, separatorChar: ',');

        // 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
        var testTrainSplit = context.Data.TrainTestSplit(trainingData, testFraction: 0.2f);

        // 4. Wybór algorytmu - w tym przypadku regresji liniowej
        var trainer = context.Regression.Trainers.LbfgsPoissonRegression();

        // 5. Budowanie modelu
        var pipeline = 
            context.Transforms.Categorical.OneHotEncoding(outputColumnName: "ModelEncoded", inputColumnName: "Model") // Kodowanie kategorii za pomocą OneHotEncoding
            .Append(context.Transforms.Concatenate("Features", nameof(CarData.Mileage), "ModelEncoded")
            .Append(context.Transforms.NormalizeMinMax("Features")) // Normalizacja danych Min-Max - przekształca wartości w zbiorze danych na przedział [0,1] lub [-1, 1]
            .Append(trainer));

        // 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
        var model = pipeline.Fit(testTrainSplit.TrainSet);

        // 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
        var predictions = model.Transform(testTrainSplit.TestSet);

        var metrics = context.Regression.Evaluate(predictions);

        Console.WriteLine($"R^2: {metrics.RSquared}"); // Współczynnik determinacji R^2
        Console.WriteLine($"MEA: {metrics.MeanAbsoluteError}"); // Średni błąd w jednostkach ceny
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}"); // Wskazuje, jak bardzo predykcje odbiegają od rzeczywistych wartości

        // 8. Predykcja dla nowego przypadku

        var newCarData = new CarData { Mileage = 50_000, Model = "Toyota" };

        // 9. Tworzenie silnika predykcyjnego
        var predictionEngine = context.Model.CreatePredictionEngine<CarData, CarPricePrediction>(model);

        // 10. Predykcja ceny samochodu a na podstawie przebiegu
        var prediction = predictionEngine.Predict(newCarData);

        // 11. Wyświetlenie wyniku predykcji
        Console.WriteLine($"Prediction: {prediction.PredictedPrice}");

        Console.ReadLine();
    }
}

public class CarData
{
    [LoadColumn(0)]
    public float Mileage;

    [LoadColumn(1), ColumnName("Label")]
    public float Price;

    [LoadColumn(2)]
    public string Model;
}


public class CarPricePrediction
{
    [ColumnName("Score")]
    public float PredictedPrice;
}