require 'csv'
require 'ruby-fann'

x_data = []
y_data = []
# Загружаем данные из CSV файла в два массива. Первый - независимые переменные X и второй массив - переменные Y, зависящие от X.
CSV.foreach("../common-data/admission.csv", headers: false) do |row|
  x_data.push([row[0].to_f, row[1].to_f])
  y_data.push([row[2].to_i])
end

# Разделим данные на наборы для тестирования и обучения
test_size_percentange = 20.0 # 20.0% на тестирование
test_set_size = x_data.size * (test_size_percentange/100.0)
test_x_data = x_data[0 .. (test_set_size - 1)]
test_y_data = y_data[0 .. (test_set_size - 1)]
training_x_data = x_data[test_set_size .. x_data.size]
training_y_data = y_data[test_set_size .. y_data.size]

# Установка модели для обучения
train = RubyFann::TrainData.new(inputs: training_x_data, desired_outputs: training_y_data)

# Установим модель и обучим её, используя данные для обучения
model = RubyFann::Standard.new(
  num_inputs: 2,
  hidden_neurons: [6],
  num_outputs: 1
)

# 5000 итераций для обучения, 500 итераций между проверкой изменения ошибки
model.train_on_data(train, 5000, 500, 0.01)

# Предсказание простого класса (1 - поступил, 0 - не поступил)
prediction = model.run([45, 85])
# Округлим выход сделанного предсказания
puts "Алгоритм предсказал класс: #{prediction.map(&:round)}"

predicted = []
test_x_data.each do |params|
  predicted.push(model.run(params).map(&:round))
end
correct = predicted.collect.with_index { |e,i| (e == test_y_data[i]) ? 1 : 0 }.inject { |sum,e| sum+e }
puts "Точность: #{(correct / test_set_size * 100.0).round(2)}% - тестовый набор данных #{test_size_percentange}%"
