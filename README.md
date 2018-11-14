### How-to

A simulation can be started with a command as the following:

```commandline
$ ./gradlew --no-daemon && \
  java -Xmx5024m -cp "build/classes/main:libs/alchemist-redist-8.0.0-beta-b76701c.jar:build/resources/main" \
  it.unibo.alchemist.Alchemist \
  -b -var smartness \
  -y src/main/yaml/casestudy.yml -e data/20181105 -t 500 -p 3 -v &> exec.txt &
```
