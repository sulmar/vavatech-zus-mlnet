using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WaterConsumptionPrediction;

public static class DateTimeExtensions
{
    public static string GetSeason(this DateTime date) => date.Month switch
    {
        12 or 1 or 2 => "Winter",
        3 or 4 or 5 => "Spring",
        6 or 7 or 8 => "Summer",
        9 or 10 or 11 => "Autumn",
        _ => throw new ArgumentOutOfRangeException()
    };

    // Przyjmujemy, że sobota i niedziela to dni wolne od pracy.
    public static bool IsWorkingDay(this DateTime date) => date.DayOfWeek != DayOfWeek.Saturday && date.DayOfWeek != DayOfWeek.Sunday;
}
