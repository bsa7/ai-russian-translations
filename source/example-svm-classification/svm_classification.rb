# frozen_string_literal: true

require 'csv'
require 'libsvm'

x_data = []
y_data = []

# Загружаем данные из CSV файла в два массива - один для независимой переменной X и один для зависимой переменной Y
CSV.foreach('./source/common-data/admission.csv', headers: false) do |row|
  x_data.push([row[0].to_f, row[1].to_f])
  y_data.push(row[2].to_i)
end

# Разделим данные на набор для тестирования и набор для обучения
test_size_percentange = 20.0 # 20.0% для тестирования
test_set_size = x_data.size * (test_size_percentange / 100.0)
test_x_data = x_data[0..(test_set_size - 1)]
test_y_data = y_data[0..(test_set_size - 1)]
training_x_data = x_data[test_set_size..x_data.size]
training_y_data = y_data[test_set_size..y_data.size]

# Установим параметры SVM
parameter = Libsvm::SvmParameter.new
parameter.cache_size = 1 # в мегабайтах
parameter.eps = 0.001
parameter.c = 1
parameter.gamma = 0.01
parameter.kernel_type = Libsvm::KernelType::RBF

# Преобразование в векторы LibSVM
test_x_data = test_x_data.map { |feature_row| Libsvm::Node.features(feature_row) }
training_x_data = training_x_data.map { |feature_row| Libsvm::Node.features(feature_row) }

# Определим нашу проблему, используя данные для обучения
problem = Libsvm::Problem.new
problem.set_examples(training_y_data, training_x_data)

# Обучим модель
model = Libsvm::Model.train(problem, parameter)

# Предсказываем простой класс
prediction = model.predict( Libsvm::Node.features([45, 85]))
# Округлим вывод для осуществления предсказания
puts "Алгоритм предсказал класс: #{prediction}"

predicted = []
test_x_data.each do |params|
  predicted.push(model.predict(params))
end
correct = predicted.collect.with_index { |e, i| e == test_y_data[i] ? 1 : 0 }.inject { |sum, e| sum + e }
puts [
  'Точность классификации:',
  "#{((correct.to_f / test_set_size) * 100).round(2)}%",
  '-',
  'Доля тестового набора',
  "#{test_size_percentange}%"
].join(' ')
