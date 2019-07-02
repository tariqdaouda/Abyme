# Abyme (Fractals)

Abyme is a tool for writing Deep and Sophisticated (Training) Loops.

Training loops involve a lot cuisine:
* When to save a model
* What to print on screen
* When?
* What information capture for debugging
* In what format save them?
* At which periodicity?

With Abyme training loops are written as fractals that go deeper and deeper, allowing the user to dynamically plug events at *user-defined* steps. Sounds complicated but it actually makes everyting much simpler.

```python
    criterion = torch.nn.modules.loss.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    epoch_looper = AB.IterrationLooper()

    train_data_looper = AB.DataLooper(get_data_loader(train=True, mask_targets=True, batch_size=500))
    train_pass = AP.SupervisedPass(model, optimizer, criterion, update_parameters=True, inputs_targets_formater=data_formater)
    train_stats = AB.Stats(caller_field="last_loss")

    test_data_looper = AB.DataLooper(get_data_loader(train=False, mask_targets=True, batch_size=10000))
    test_pass = AP.SupervisedPass(model, optimizer, criterion, update_parameters=False, inputs_targets_formater=data_formater)
    test_stats = AB.Stats(caller_field="last_loss")

    csv_result = AB.CSVWriter(filename="test2.csv")

    def handle_epoch_end(name, epoch_looper, data_looper, csv, save_model, stats_caller_focus):
        res = (
            AB.NewLowTrigger("average").focus(stats_caller_focus)("dig",
                AB.Print(["==>New %s average low, epoch"%name, epoch_looper.get('counter'), "batch:", data_looper.get("counter")]),
                AB.If(condition=save_model)("dig",
                    AP.SaveModel(model=model, filename=name, prefix=epoch_looper.get("counter")),
                ),
                AB.PrettyPrintStore(fields=["average", "std", "min", "max"], prefix="%s.new.low." % name),
                csv.add_caller_to_line(fields=["average", "std", "min", "max"], prefix="%s.new.low." % name),
            ),
            AB.MovingStats("average", window_size=100).focus(stats_caller_focus)("dig",
                AB.PeriodicTrigger(100, wait_periods=1)("dig",
                    AB.PrettyPrintStore(fields=["average", "std", "min", "max"], prefix="%s.loss.moving." % name),
                    csv.add_caller_to_line(fields=["average", "std", "min", "max"], prefix="%s.loss.moving." % name),
                )
            ),
        )
        return res 
    
    AB.Ground()("dig",
        epoch_looper.setup(10)("start",
            AB.Print(["Training starts"])
        ).at("iteration_start",
            csv_result.open_line(),
            train_data_looper("iteration_end",
                train_pass("end",
                    train_stats,
                )
            ).at("end",
                *handle_epoch_end("train", epoch_looper, train_data_looper, csv_result, save_model=True, stats_caller_focus=train_stats),
                test_data_looper("iteration_end",
                    test_pass("end",
                        test_stats,
                        *handle_epoch_end("test", epoch_looper, test_data_looper, csv_result, save_model=True, stats_caller_focus=test_stats)
                    ),
                ),
            )
        ).at("iteration_start", 
            csv_result.commit_line(),
            csv_result.save(),
            test_stats.reset,
            train_stats.reset
        ).at("end",
           AB.Print("End of training")
        )
    ).dig()
```
