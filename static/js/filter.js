document.getElementById('apply-filter').addEventListener('click', function () {
    const selectedUndertones = Array.from(
      document.querySelectorAll('input[name="undertone"]:checked')
    ).map(input => input.value);
  
    const selectedPrices = Array.from(
      document.querySelectorAll('input[name="price"]:checked')
    ).map(input => input.value);
  
    const selectedStatus = Array.from(
      document.querySelectorAll('input[name="status"]:checked')
    ).map(input => input.value);
  
    const products = document.querySelectorAll('.product-card');
  
    products.forEach(product => {
      const undertone = product.dataset.undertone;
      const price = product.dataset.price;
      const status = product.dataset.status;
  
      const matchesUndertone = selectedUndertones.length === 0 || selectedUndertones.includes(undertone);
      const matchesPrice = selectedPrices.length === 0 || selectedPrices.includes(price);
      const matchesStatus = selectedStatus.length === 0 || selectedStatus.includes(status);
  
      if (matchesUndertone && matchesPrice && matchesStatus) {
        product.style.display = 'block';
      } else {
        product.style.display = 'none';
      }
    });
  });
  