require 'csv'
require 'liblinear'

x_data = []
y_data = []

# Загружаем данные из CSV файла в два массива. Первый - независимые переменные X и второй массив - переменные Y, зависящие от X.
file_name = '../common-data/data/admission.csv'
CSV.foreach(file_name, :headers => false) do |row|
  x_data.push( [row[0].to_f, row[1].to_f] )
  y_data.push( row[2].to_i )
end

# Разделение данных на два набора - для обучения и тестирования.
test_size_percentange = 20.0 # 20.0%
test_set_size = x_data.size * (test_size_percentange / 100.0)
test_x_data = x_data[0...test_set_size]
test_y_data = y_data[0...test_set_size]
training_x_data = x_data[test_set_size..x_data.size]
training_y_data = y_data[test_set_size..y_data.size]

model = Liblinear.train(
  { solver_type: Liblinear::L2R_LR },   # Решающее устройство: L2R_LR - L2-упорядоченная логистическая регрессия
  training_y_data,                      # Данные для обучения - результаты классификации
  training_x_data,                      # Данные для обучения - результаты экзаменов
  100                                   # Смещение (диапазон исходных данных)
)

# Предсказываемый класс
prediction = Liblinear.predict(model, [45, 85])
# Получить вероятности попадания в класс
probs = Liblinear.predict_probabilities(model, [45, 85])
probs = probs.sort
puts "Алгоритм предсказал класс #{prediction}"
puts "#{(probs[1] * 100).round(2)}% возможность попадания в целевой класс"
puts "#{(probs[0] * 100).round(2)}% возможность попадания в другие классы"

predicted = []
test_x_data.each do |params|
  predicted.push(Liblinear.predict(model, params))
end
correct = predicted.collect.with_index { |predict, index| predict == test_y_data[index] ? 1 : 0 }.inject { |sum, e| sum + e }
puts "Точность: #{(correct.to_f / test_set_size * 100).round(2)}% - размер тестового набора #{test_size_percentange}%"
