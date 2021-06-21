require 'csv'
require 'ruby_linear_regression'

x_data = []
y_data = []
# Загружаем данные из CSV file в два массива - один для независимой переменной X и один для зависимой переменной Y
# Каждая строка файла содержит площадь участка и жилую площадь:
# [SQ_FEET_PROPERTY, SQ_FEET_HOUSE]
file_name = './source/example-linear-regression/data/staten-island-single-family-home-sales-2015.csv'
CSV.foreach(file_name, headers: true) do |row|
  x_data.push([row[0].to_i, row[1].to_i])
  y_data.push(row[2].to_i)
end

## Пример 1 - обучение на нормальном распределении (Быстрое)
# Создаём экземпляр регрессионной модели
linear_regression = RubyLinearRegression.new
# Загружаем данные для обучения
linear_regression.load_training_data(x_data, y_data)

# Обучим модель, используя нормальное распределение
linear_regression.train_normal_equation

## Пример 2 - обучение на градиентном спуске (Медленное)

# Создание регрессионной модели
linear_regression_gradient_descent = RubyLinearRegression.new

# Загрузка данных для обучения
linear_regression_gradient_descent.load_training_data(x_data, y_data)

linear_regression_gradient_descent.train_gradient_descent(0.0005, 500, true)

# Вывод цен для сравнения
puts "Цена - нормальное распределение: #{linear_regression.compute_cost}"
puts "Цена - градиентный спуск: #{linear_regression_gradient_descent.compute_cost}"

# Вывод предсказаний

# Нормальное распределение
# Предсказать цену жилого дома с площадью участка 2000 кв. футов и жилой площадью 1500 кв. футов
prediction_data = [2000, 1500]
predicted_price = linear_regression.predict(prediction_data.clone)
puts [
  'С помощью линейной регрессии предсказана цена продажи для дома',
  "с площадью участка #{prediction_data[0]} кв. футов",
  "и жилой площадью #{prediction_data[1]} кв.футов:",
  "$#{predicted_price.round}."
].join(' ')

# Градиентный спуск
# Предсказать цену жилого дома с площадью участка 2000 кв. футов и жилой площадью 1500 кв. футов
prediction_data = [2000, 1500]
predicted_price = linear_regression_gradient_descent.predict(prediction_data.clone)
puts [
  'С помощью градиентной линейной регрессии предсказана цена продажи для дома',
  "с площадью участка #{prediction_data[0]} кв. футов",
  "и жилой площадью #{prediction_data[1]} кв.футов:",
  "$#{predicted_price.round}."
].join(' ')
