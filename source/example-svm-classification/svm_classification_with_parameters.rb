# frozen_string_literal: true

require 'csv'
require 'libsvm'

x_data = []
y_data = []

# Загружаем данные из CSV файла в два массива - один для независимой переменной X и один для зависимой переменной Y
CSV.foreach('./source/common-data/admission.csv', headers: false) do |row|
  x_data.push [row[0].to_f, row[1].to_f]
  y_data.push row[2].to_i
end

# Разбиваем данные на наборы для обучения, проверки и тестирования
validation_size_percentange = 15.0 # 15%
validation_set_size = x_data.size * validation_size_percentange / 100.to_f
test_size_percentange = 15.0 # 20%
test_set_size = x_data.size * (test_size_percentange / 100.to_f)
validation_x_data = x_data[0..(validation_set_size - 1)]
validation_y_data = y_data[0..(validation_set_size - 1)]
test_x_data = x_data[validation_set_size..(validation_set_size+test_set_size - 1)]
test_y_data = y_data[validation_set_size..(validation_set_size+test_set_size - 1)]
training_x_data = x_data[(validation_set_size + test_set_size)..x_data.size]
training_y_data = y_data[(validation_set_size + test_set_size)..y_data.size]

# Convert into proper feature arrays for Libsvm
validation_x_data = validation_x_data.map { |feature_row| Libsvm::Node.features(feature_row) }
test_x_data = test_x_data.map { |feature_row| Libsvm::Node.features(feature_row) }
training_x_data = training_x_data.map { |feature_row| Libsvm::Node.features(feature_row) }

# Определяем нашу проблему, используя данные для обучения
problem = Libsvm::Problem.new
problem.set_examples(training_y_data, training_x_data)

posible_values = [0.0001, 0.0005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
best_c = 0
best_gamma = 0
best_accuracy = 0

posible_values.each do |test_c|
  posible_values.each do |test_gamma|
    parameter = Libsvm::SvmParameter.new
    parameter.cache_size = 1 # в megabytes
    parameter.eps = 0.001
    parameter.gamma = test_gamma
    parameter.c = test_c
    parameter.kernel_type = Libsvm::KernelType::RBF

    # Обучение модели
    model = Libsvm::Model.train(problem, parameter)
    predicted = []
    validation_x_data.each do |params|
      predicted.push( model.predict(params) )
    end

    correct = predicted.collect.with_index { |e, i| e == validation_y_data[i] ? 1 : 0 }.sum
    accuracy = ((correct.to_f / validation_set_size) * 100).round(2)
    next if accuracy <= best_accuracy

    best_accuracy = accuracy
    best_c = test_c
    best_gamma = test_gamma
    puts "Новое улучшение! Точность классификации: #{accuracy}% - C=#{test_c}, гамма=#{test_gamma}"
  end
end

# Установка оптимальных параметров для модели и вычисление точности классификации
parameter = Libsvm::SvmParameter.new
parameter.cache_size = 1 # в megabytes
parameter.eps = 0.001
parameter.gamma = best_gamma
parameter.c = best_c
parameter.kernel_type = Libsvm::KernelType::RBF

# Обучение модели
model = Libsvm::Model.train(problem, parameter)
predicted = []
test_x_data.each do |params|
  predicted.push( model.predict(params) )
end

correct = predicted.collect.with_index { |e, i| e == test_y_data[i] ? 1 : 0 }.sum
accuracy = ((correct.to_f / test_set_size) * 100).round(2)

puts "Итоговая точность классификации: #{accuracy}% - C=#{best_c}, гамма=#{best_gamma}"
