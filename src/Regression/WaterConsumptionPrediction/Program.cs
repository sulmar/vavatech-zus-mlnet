using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;
using WaterConsumptionPrediction;

Console.WriteLine("Hello, Water Consumption Prediction!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy surowe dane z pliku CSV
var rawData = context.Data.LoadFromTextFile<DeliveryRawData>("water_consumption.csv", hasHeader: false, separatorChar: ';');

var preview = rawData.Preview();

// 3. Konwertujemy kolumnę z datą ręcznie
var deliveries = context.Data.CreateEnumerable<DeliveryRawData>(rawData, reuseRowObject: false)
    .Select(row => {
        
        var date = DateTime.ParseExact(row.DeliveryDate, "dd.MM.yyyy", CultureInfo.InvariantCulture);
        
        return new DeliveryData
        {
            CustomerId = row.CustomerId,
            DeliveryDate = date,
            OrderSize = row.OrderSize,
            Season = date.GetSeason(),
            IsWorkingDay = date.IsWorkingDay(),
            DayOfWeek = (float)date.DayOfWeek,
            Month = (float)date.Month
        };
    })
    .ToList();

var data = deliveries
          .GroupBy(d => d.CustomerId)
          .SelectMany(group =>
          {
              var sorted = group.OrderBy(d => d.DeliveryDate).ToList();
              return sorted.Zip(sorted.Skip(1), (prev, curr) => new DeliveryData
              {
                  CustomerId = curr.CustomerId,
                  DeliveryDate = curr.DeliveryDate,
                  OrderSize = curr.OrderSize,
                  Season = curr.Season,
                  IsWorkingDay = curr.IsWorkingDay,
                  DaysSinceLastDelivery = (curr.DeliveryDate - prev.DeliveryDate).Days,
                  OrderSizeLastDelivery = prev.OrderSize
              });

          });

var trainingData = context.Data.LoadFromEnumerable(data);

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainingData, testFraction: 0.2f);

// 4. Wybór algorytmu - FastTree
// dotnet add package Microsoft.ML.FastTree

var trainer = context.Regression.Trainers.FastTree();

// 5. Budowanie modelu
var pipeline = context.Transforms.Conversion.MapValueToKey("CustomerKey", "CustomerId")
    //.Append(context.Transforms.NormalizeMinMax("DaysSinceLastDelivery"))
    //.Append(context.Transforms.NormalizeMinMax("OrderSizeLastDelivery"))
    .Append(context.Transforms.Categorical.OneHotEncoding("EncodedCustomer", "CustomerKey"))
    .Append(context.Transforms.Conversion.MapValueToKey("SeasonKey", "Season"))
    .Append(context.Transforms.Categorical.OneHotEncoding("EncodedSeason", "SeasonKey"))
    .Append(context.Transforms.Conversion.ConvertType("IsWorkingDay", outputKind: DataKind.Single))
    .Append(context.Transforms.Concatenate("Features", "EncodedCustomer", "EncodedSeason", "DaysSinceLastDelivery", "IsWorkingDay", "DayOfWeek", "Month", "OrderSizeLastDelivery"))
    .Append(trainer);


// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(testTrainSplit.TrainSet);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testTrainSplit.TestSet);

var metrics = context.Regression.Evaluate(predictions);

Console.WriteLine($"R^2: {metrics.RSquared:P2}"); // Współczynnik determinacji R^2
Console.WriteLine($"MEA: {metrics.MeanAbsoluteError}"); // Średni błąd bezwzględny (to średnia różnica między przewidywaną a rzeczywistą wartością, co pozwala oszacować, jak bardzo prognozy odbiegają od obserwowanych danych.)
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}"); // Wskazuje, jak bardzo predykcje odbiegają od rzeczywistych wartości


Console.ReadLine();


public class DeliveryRawData
{
    [LoadColumn(0)]
    public float CustomerId { get; set; }

    [LoadColumn(1)]
    public string DeliveryDate { get; set; }

    // Wartość, którą chcemy przewidzieć
    [LoadColumn(2)]
    public float OrderSize { get; set; }
}

// Klasa docelowa po konwersji daty
public class DeliveryData
{
    public float CustomerId { get; set; }
    public DateTime DeliveryDate { get; set; }

    [ColumnName("Label")]
    public float OrderSize { get; set; }

    public string Season { get; set; }
    public bool IsWorkingDay { get; set; }
    public float DaysSinceLastDelivery { get; set; }

    public float DayOfWeek { get; set; }  // Dodajemy dzień tygodnia
    public float Month { get; set; }      // Dodajemy miesiąc


    public float OrderSizeLastDelivery { get; set; }
}

public class DeliveryPrediction
{
    [ColumnName("Score")]
    public float PredictedOrderSize { get; set; }
}
