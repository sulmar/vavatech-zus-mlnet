using Microsoft.ML;
using Microsoft.ML.Data;

internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");

        // dotnet add package Microsoft.ML

        // 1. Tworzymy kontekst ML.NET
        var context = new MLContext();

        // Przykładowe dane z Polski
        var data = new[]
        {
            new HouseData { Size = 65f, Location = "Warszawa", Bedrooms = 2f, Price = 650000f },
            new HouseData { Size = 80f, Location = "Kraków", Bedrooms = 3f, Price = 720000f },
            new HouseData { Size = 50f, Location = "Łódź", Bedrooms = 1f, Price = 300000f },
            new HouseData { Size = 90f, Location = "Wrocław", Bedrooms = 3f, Price = 680000f },
            new HouseData { Size = 70f, Location = "Gdańsk", Bedrooms = 2f, Price = 550000f },
            new HouseData { Size = 120f, Location = "Poznań", Bedrooms = 4f, Price = 900000f },
            new HouseData { Size = 45f, Location = "Katowice", Bedrooms = 1f, Price = 280000f },
            new HouseData { Size = 85f, Location = "Warszawa", Bedrooms = 3f, Price = 850000f },
            new HouseData { Size = 60f, Location = "Kraków", Bedrooms = 2f, Price = 540000f },
            new HouseData { Size = 75f, Location = "Rzeszów", Bedrooms = 2f, Price = 450000f },
            // Nowe dane:
            new HouseData { Size = 55f, Location = "Warszawa", Bedrooms = 1f, Price = 480000f },
            new HouseData { Size = 95f, Location = "Kraków", Bedrooms = 4f, Price = 880000f },
            new HouseData { Size = 40f, Location = "Łódź", Bedrooms = 1f, Price = 250000f },
            new HouseData { Size = 100f, Location = "Wrocław", Bedrooms = 3f, Price = 750000f },
            new HouseData { Size = 82f, Location = "Gdańsk", Bedrooms = 3f, Price = 620000f },
            new HouseData { Size = 110f, Location = "Poznań", Bedrooms = 4f, Price = 870000f },
            new HouseData { Size = 38f, Location = "Katowice", Bedrooms = 1f, Price = 240000f },
            new HouseData { Size = 78f, Location = "Warszawa", Bedrooms = 2f, Price = 700000f },
            new HouseData { Size = 68f, Location = "Kraków", Bedrooms = 2f, Price = 580000f },
            new HouseData { Size = 62f, Location = "Rzeszów", Bedrooms = 2f, Price = 420000f },
            new HouseData { Size = 130f, Location = "Warszawa", Bedrooms = 5f, Price = 1200000f },
            new HouseData { Size = 72f, Location = "Łódź", Bedrooms = 3f, Price = 460000f },
            new HouseData { Size = 88f, Location = "Wrocław", Bedrooms = 3f, Price = 710000f },
            new HouseData { Size = 58f, Location = "Gdańsk", Bedrooms = 2f, Price = 490000f },
            new HouseData { Size = 105f, Location = "Poznań", Bedrooms = 4f, Price = 830000f },
            new HouseData { Size = 52f, Location = "Katowice", Bedrooms = 2f, Price = 340000f },
            new HouseData { Size = 92f, Location = "Kraków", Bedrooms = 3f, Price = 760000f },
            new HouseData { Size = 66f, Location = "Rzeszów", Bedrooms = 2f, Price = 430000f },
            new HouseData { Size = 77f, Location = "Warszawa", Bedrooms = 3f, Price = 780000f },
            new HouseData { Size = 48f, Location = "Łódź", Bedrooms = 1f, Price = 290000f }
        };

        // 2.Wczytujemy dane z kolekcji
        var trainingData = context.Data.LoadFromEnumerable(data);

        // 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
        var testTrainSplit = context.Data.TrainTestSplit(trainingData, testFraction: 0.2f);

        // 4. Wybór algorytmu - w tym przypadku regresji liniowej
        var trainer = context.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Price");

        // 5. Budowanie modelu
        var pipeline = context.Transforms.Categorical.OneHotEncoding("LocationEncoded", "Location")
            .Append(context.Transforms.Concatenate("Features", "Size", "LocationEncoded", "Bedrooms"))
            .Append(trainer);


        // 6. Trenowanie modelu na danych treningowych
        var model = pipeline.Fit(testTrainSplit.TrainSet);

        // 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
        var predictions = model.Transform(testTrainSplit.TestSet);

        var metrics = context.Regression.Evaluate(predictions, labelColumnName: "Price");

        Console.WriteLine($"R^2: {metrics.RSquared}"); // Współczynnik determinacji R^2
        Console.WriteLine($"MEA: {metrics.MeanAbsoluteError}"); // Średni błąd w jednostkach ceny
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}"); // Wskazuje, jak bardzo predykcje odbiegają od rzeczywistych wartości

        // 8. Predykcja dla nowego przypadku
        var newHouseData = new HouseData
        {
            Size = 70f,
            Location = "Warszawa",
            Bedrooms = 2f
        };

        // 9. Tworzenie silnika predykcyjnego
        var predictionEngine = context.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);

        // 10. Predykcja ceny samochodu a na podstawie przebiegu
        var prediction = predictionEngine.Predict(newHouseData);

        // 11. Wyświetlenie wyniku predykcji
        Console.WriteLine($"Prediction: {prediction.PredictedPrice}");

        Console.ReadLine();
    }
}

public class HouseData
{
    public float Size { get; set; }
    public string Location { get; set; }
    public float Bedrooms { get; set; }
    public float Price { get; set; }
}

public class HousePrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}