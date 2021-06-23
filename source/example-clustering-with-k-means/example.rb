# frozen_string_literal: true

require 'csv'
require 'colorize'
require 'kmeans-clusterer'

data = []
labels = []
# Загружаем данные из CSV файла в два массива - один для координат широты и долготы, а второй для названий городов
CSV.foreach('./source/common-data/california-cities.csv', headers: true) do |row|
  labels.push row[1]
  data.push [row[2].to_f, row[3].to_f]
end

k = 3 # Искомое количество кластеров
kmeans = KMeansClusterer.run k, data, labels: labels, runs: 100

kmeans.clusters.each do |cluster|
  puts "#{"\nКластер: ".green} #{cluster.id}"
  puts "#{'Центр кластера: '.yellow}#{cluster.centroid}"
  puts "#{'Города кластера: '.yellow}#{cluster.points.map(&:label).join(', ')}"
end

2.upto(20) do |count_of_clusters|
  kmeans = KMeansClusterer.run count_of_clusters, data, labels: labels, runs: 100
  puts "Количество кластеров: #{count_of_clusters}, Ошибка: #{kmeans.error.round(2)}"
end

k = 6 # Оптимальное k найдено с использованием метода локтя
kmeans = KMeansClusterer.run k, data, labels: labels, runs: 1000
kmeans.clusters.each do |cluster|
 puts "Кластер № #{cluster.id}"
 puts "Центр кластера: #{cluster.centroid}"
 puts "Города кластера: " + cluster.points.map(&:label).join(', ')
end

city_clusters = []
kmeans.clusters.each do |cluster|
  cluster.points.each do |city|
    city_clusters.push "#{city.label}, #{city.data.to_a.join(', ')}, #{cluster.id}"
  end
end

puts city_clusters.sort
