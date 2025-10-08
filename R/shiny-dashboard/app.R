# app.R
library(shiny)
library(tidyverse)

ui <- fluidPage(
  titlePanel("mtcars explorer"),
  sidebarLayout(
    sidebarPanel(
      checkboxGroupInput(
        "cyl", "Cylinders",
        choices = sort(unique(mtcars$cyl)),
        selected = sort(unique(mtcars$cyl))
      ),
      sliderInput("wt_range", "Weight range",
                  min = min(mtcars$wt), max = max(mtcars$wt),
                  value = c(min(mtcars$wt), max(mtcars$wt)), step = 0.1),
      checkboxInput("show_lm", "Show linear fit", value = TRUE)
    ),
    mainPanel(
      plotOutput("scatter", height = "380px"),
      tags$hr(),
      verbatimTextOutput("model_summary")
    )
  )
)

server <- function(input, output, session) {
  data_f <- reactive({
    mtcars %>%
      rownames_to_column("model") %>%
      filter(cyl %in% input$cyl, wt >= input$wt_range[1], wt <= input$wt_range[2])
  })

  output$scatter <- renderPlot({
    d <- data_f()
    p <- ggplot(d, aes(wt, mpg, label = model)) +
      geom_point() +
      ggrepel::geom_text_repel(size = 3, max.overlaps = 10) +
      labs(x = "Weight (1000 lbs)", y = "MPG")
    if (input$show_lm && nrow(d) > 1) {
      p <- p + geom_smooth(method = "lm", se = FALSE)
    }
    p + theme_minimal()
  })

  output$model_summary <- renderPrint({
    d <- data_f()
    if (nrow(d) > 1) {
      m <- lm(mpg ~ wt, data = d)
      summary(m)
    } else {
      cat("Not enough data to fit a model.")
    }
  })
}

shinyApp(ui, server)