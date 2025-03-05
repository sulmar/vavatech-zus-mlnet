
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, Salary Prediction!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane z pliku CSV
var trainingData = context.Data.LoadFromTextFile<SalaryData>("salary-data.csv", hasHeader: true, separatorChar: ',');

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainingData, testFraction: 0.2f);

// 4. Wybór algorytmu - wybieram Catalog klasyfikacji regresji liniowej
var trainer = context.Regression.Trainers.LbfgsPoissonRegression();

// 5. Budowanie modelu
var pipeline = 
    context.Transforms.Concatenate("Features", "YearsExperience")  // Łączymy dane wejściowe w cechy
    .Append(trainer);

// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(testTrainSplit.TrainSet);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testTrainSplit.TestSet);

var metrics = context.Regression.Evaluate(predictions);

Console.WriteLine($"R^2: {metrics.RSquared}"); // Współczynnik determinacji R^2
Console.WriteLine($"MEA: {metrics.MeanAbsoluteError}"); // Średni błąd w jednostkach ceny
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}"); // Wskazuje, jak bardzo predykcje odbiegają od rzeczywistych wartości

// 8. Predykcja dla nowego przypadku
var newSalaryData = new SalaryData { YearsExperience = 1.1f };

// 9. Tworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<SalaryData, SalaryPrediction>(model);

// 10. Predykcja wynagrodzenia na podstawie lat doświadczenia
var prediction = predictionEngine.Predict(newSalaryData);

// 11. Wyświetlenie wyniku predykcji
Console.WriteLine($"Prediction: {prediction.PredictedSalary}");

Console.ReadLine();

// Klasa reprezentuje dane wejściowe dla modelu
public class SalaryData
{
    [LoadColumn(0)]  // Atrybut [LoadColumn(n)] określa, z której kolumny powinny być wczytane wartości
    public float YearsExperience;

    [LoadColumn(1), ColumnName("Label")] // Etykieta (label), która oznacza, że model będzie się uczyć przewidywania tej wartości
    public float Salary;
}


// Klasa reprezentuje wynik działania modelu
public class SalaryPrediction
{
    [ColumnName("Score")]
    public float PredictedSalary { get; set; }
}
