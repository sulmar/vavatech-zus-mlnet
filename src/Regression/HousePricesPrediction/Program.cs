using Microsoft.ML;
using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations.Schema;
using static System.Runtime.InteropServices.JavaScript.JSType;

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
    new HouseData { Size = 75f, Location = "Rzeszów", Bedrooms = 2f, Price = 450000f }
};

// 2.Wczytujemy dane z kolekcji
var trainingData = context.Data.LoadFromEnumerable(data);








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
    public float PredictatedPrice { get; set; }
}